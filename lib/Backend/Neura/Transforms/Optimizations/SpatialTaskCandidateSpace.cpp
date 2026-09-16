//===- SpatialTaskCandidateSpace.cpp -----------------------------------===//
//
// Implements construction and traversal of the static rectangular
// task-shape candidate space.
//
//===----------------------------------------------------------------------===//

#include "SpatialTaskCandidateSpace.h"

#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_candidates {

// Converts the physical CGRA rectangle into the string stored in the
// Taskflow `cgra_shape` attribute, such as `1x2`.
std::string RectShape::toCgraShapeAttrValue() const {
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << llvm::formatv("{0}x{1}", rows, cols);
  return result;
}

// Returns the occupied (column, row) offsets for one valid shape. This helper
// deliberately belongs to the spatial candidate space: the shared scheduler
// only needs to place already selected tasks and should not own candidate
// packing.
static std::optional<SmallVector<std::pair<int, int>>>
getShapeCells(const CgraShape &shape, int grid_rows, int grid_cols) {
  if (shape.rows <= 0 || shape.cols <= 0 || shape.rows > grid_rows ||
      shape.cols > grid_cols) {
    return std::nullopt;
  }

  SmallVector<std::pair<int, int>> cells;
  if (shape.is_rectangular) {
    for (int row = 0; row < shape.rows; ++row) {
      for (int col = 0; col < shape.cols; ++col) {
        cells.push_back({col, row});
      }
    }
    return cells;
  }
  if (shape.cgra_positions.empty()) {
    return std::nullopt;
  }

  DenseSet<int64_t> seen;
  for (auto [col, row] : shape.cgra_positions) {
    if (col < 0 || col >= shape.cols || row < 0 || row >= shape.rows) {
      return std::nullopt;
    }
    int64_t key = static_cast<int64_t>(row) * shape.cols + col;
    if (!seen.insert(key).second) {
      return std::nullopt;
    }
    cells.push_back({col, row});
  }
  return cells;
}

static bool placeShapeChoices(size_t task_index,
                              ArrayRef<SmallVector<CgraShape>> shape_choices,
                              int grid_rows, int grid_cols,
                              MutableArrayRef<uint8_t> occupied) {
  if (task_index == shape_choices.size()) {
    return true;
  }

  for (const CgraShape &shape : shape_choices[task_index]) {
    std::optional<SmallVector<std::pair<int, int>>> cells =
        getShapeCells(shape, grid_rows, grid_cols);
    if (!cells) {
      continue;
    }
    for (int origin_row = 0; origin_row + shape.rows <= grid_rows;
         ++origin_row) {
      for (int origin_col = 0; origin_col + shape.cols <= grid_cols;
           ++origin_col) {
        bool overlaps = llvm::any_of(*cells, [&](auto offset) {
          auto [col, row] = offset;
          return occupied[static_cast<size_t>((origin_row + row) * grid_cols +
                                              origin_col + col)] != 0;
        });
        if (overlaps) {
          continue;
        }

        for (auto [col, row] : *cells) {
          occupied[static_cast<size_t>((origin_row + row) * grid_cols +
                                       origin_col + col)] = 1;
        }
        if (placeShapeChoices(task_index + 1, shape_choices, grid_rows,
                              grid_cols, occupied)) {
          return true;
        }
        for (auto [col, row] : *cells) {
          occupied[static_cast<size_t>((origin_row + row) * grid_cols +
                                       origin_col + col)] = 0;
        }
      }
    }
  }
  return false;
}

// Checks exact simultaneous placement for fixed-rotation spatial shapes.
static bool canShapesFitOnGrid(ArrayRef<CgraShape> task_shapes, int grid_rows,
                               int grid_cols) {
  if (grid_rows <= 0 || grid_cols <= 0 ||
      static_cast<uint64_t>(grid_rows) * static_cast<uint64_t>(grid_cols) >
          std::numeric_limits<size_t>::max()) {
    return false;
  }

  const uint64_t grid_area = static_cast<uint64_t>(grid_rows) * grid_cols;
  uint64_t occupied_cells = 0;
  for (const CgraShape &shape : task_shapes) {
    std::optional<SmallVector<std::pair<int, int>>> cells =
        getShapeCells(shape, grid_rows, grid_cols);
    if (!cells || cells->size() > grid_area - occupied_cells) {
      return false;
    }
    occupied_cells += cells->size();
  }

  SmallVector<CgraShape> largest_first(task_shapes.begin(), task_shapes.end());
  llvm::sort(largest_first, [](const CgraShape &lhs, const CgraShape &rhs) {
    auto cell_count = [](const CgraShape &shape) {
      return shape.is_rectangular
                 ? static_cast<int64_t>(shape.rows) * shape.cols
                 : static_cast<int64_t>(shape.cgra_positions.size());
    };
    return cell_count(lhs) > cell_count(rhs);
  });
  SmallVector<SmallVector<CgraShape>> shape_choices;
  shape_choices.reserve(largest_first.size());
  for (const CgraShape &shape : largest_first) {
    shape_choices.push_back({shape});
  }

  SmallVector<uint8_t> occupied(static_cast<size_t>(grid_rows) * grid_cols, 0);
  return placeShapeChoices(0, shape_choices, grid_rows, grid_cols, occupied);
}

// Enumerates every legal static physical rectangle and derives its mapper
// dimensions from the architecture getters. The deterministic order defines
// the mixed-radix alphabet for candidate IDs.
// TODO: Extend the analytical candidate schema and its consumers to represent
// non-rectangular shapes. The analytical search intentionally enumerates only
// fixed-rotation rectangles until that contract exists end to end.
SmallVector<RectShape> enumerateStaticRectShapes(int64_t grid_rows,
                                                 int64_t grid_cols,
                                                 int64_t per_cgra_rows,
                                                 int64_t per_cgra_cols,
                                                 int64_t max_cgras_per_task) {
  // The candidate domain contains only concrete integer rectangles.
  SmallVector<RectShape> result;
  if (grid_rows <= 0 || grid_cols <= 0 || per_cgra_rows <= 0 ||
      per_cgra_cols <= 0 || max_cgras_per_task <= 0) {
    return result;
  }

  const int64_t grid_size =
      grid_rows > std::numeric_limits<int64_t>::max() / grid_cols
          ? std::numeric_limits<int64_t>::max()
          : grid_rows * grid_cols;
  for (int64_t count = 1; count <= std::min(grid_size, max_cgras_per_task);
       ++count) {
    for (const CgraShape &physical_shape :
         taskflow::AnalyticalBasedTaskOrchestration::getRectangularShapes(
             static_cast<int>(count), static_cast<int>(grid_rows),
             static_cast<int>(grid_cols))) {
      if (physical_shape.rows >
              std::numeric_limits<int64_t>::max() / per_cgra_rows ||
          physical_shape.cols >
              std::numeric_limits<int64_t>::max() / per_cgra_cols) {
        continue;
      }
      int64_t mapper_rows = physical_shape.rows * per_cgra_rows;
      int64_t mapper_cols = physical_shape.cols * per_cgra_cols;
      result.push_back(
          {physical_shape.rows, physical_shape.cols, mapper_rows, mapper_cols});
    }
  }
  return result;
}

bool ConcurrentPackingCache::canPack(ArrayRef<RectShape> shapes) {
  Key key;
  key.reserve(shapes.size());
  for (const RectShape &shape : shapes) {
    key.push_back({shape.rows, shape.cols});
  }
  llvm::sort(key, [](const auto &lhs, const auto &rhs) {
    const int64_t lhs_area = lhs.first * lhs.second;
    const int64_t rhs_area = rhs.first * rhs.second;
    if (lhs_area != rhs_area) {
      return lhs_area > rhs_area;
    }
    if (std::max(lhs.first, lhs.second) != std::max(rhs.first, rhs.second)) {
      return std::max(lhs.first, lhs.second) > std::max(rhs.first, rhs.second);
    }
    return lhs > rhs;
  });

  auto found = results_.find(key);
  if (found != results_.end()) {
    return found->second;
  }

  bool result = false;
  if (grid_rows_ > 0 && grid_cols_ > 0 &&
      grid_rows_ <= std::numeric_limits<int>::max() &&
      grid_cols_ <= std::numeric_limits<int>::max()) {
    SmallVector<CgraShape> fixed_shapes;
    fixed_shapes.reserve(shapes.size());
    bool dimensions_fit = true;
    for (const RectShape &shape : shapes) {
      if (shape.rows <= 0 || shape.cols <= 0 ||
          shape.rows > std::numeric_limits<int>::max() ||
          shape.cols > std::numeric_limits<int>::max()) {
        dimensions_fit = false;
        break;
      }
      fixed_shapes.push_back({static_cast<int>(shape.rows),
                              static_cast<int>(shape.cols),
                              true,
                              {}});
    }
    if (dimensions_fit) {
      result = canShapesFitOnGrid(fixed_shapes, static_cast<int>(grid_rows_),
                                  static_cast<int>(grid_cols_));
    }
  }
  results_.emplace(std::move(key), result);
  return result;
}

bool visitConcurrentlyPackableShapeTuples(
    ArrayRef<SmallVector<RectShape>> shapes_by_task,
    ConcurrentPackingCache &packing, ShapeIndexTupleConsumer consume) {
  const int64_t grid_rows = packing.gridRows();
  const int64_t grid_cols = packing.gridCols();
  if (shapes_by_task.empty() || grid_rows <= 0 || grid_cols <= 0 ||
      llvm::any_of(shapes_by_task,
                   [](const auto &shapes) { return shapes.empty(); }) ||
      grid_rows > std::numeric_limits<int64_t>::max() / grid_cols) {
    return true;
  }
  const int64_t grid_area = grid_rows * grid_cols;
  // Every supported task consumes at least one physical CGRA. This prevents a
  // large task list from expanding an obviously empty Cartesian product.
  if (shapes_by_task.size() > static_cast<size_t>(grid_area)) {
    return true;
  }

  SmallVector<size_t> selected_indices;
  SmallVector<RectShape> selected_shapes;
  uint64_t valid_index = 0;
  std::function<bool(size_t, int64_t)> visit = [&](size_t task_index,
                                                   int64_t selected_area) {
    if (task_index == shapes_by_task.size()) {
      if (!packing.canPack(selected_shapes)) {
        return true;
      }
      if (!consume(valid_index, selected_indices)) {
        return false;
      }
      ++valid_index;
      return true;
    }

    // Shapes remain in their declared deterministic order. The area test
    // removes only tuples that cannot possibly fit; geometry is checked
    // exactly after one shape has been chosen for every task.
    for (auto [shape_index, shape] :
         llvm::enumerate(shapes_by_task[task_index])) {
      const int64_t area = shape.cgraCount();
      if (area <= 0 || area > grid_area - selected_area) {
        continue;
      }
      selected_indices.push_back(shape_index);
      selected_shapes.push_back(shape);
      if (!visit(task_index + 1, selected_area + area)) {
        return false;
      }
      selected_shapes.pop_back();
      selected_indices.pop_back();
    }
    return true;
  };
  return visit(0, 0);
}

// Serializes the shape fields used by candidate records. The explicit tile
// dimensions are the only mapper-shape truth; a `rect-4x8` string is produced
// only for diagnostics when needed.
static llvm::json::Object shapeJson(const RectShape &shape) {
  llvm::json::Object object;
  object["kind"] = "rect";
  object["rows"] = shape.rows;
  object["cols"] = shape.cols;
  object["cgra_count"] = shape.cgraCount();
  object["cgra_shape"] = shape.toCgraShapeAttrValue();
  object["mapper_tile_rows"] = shape.mapper_rows;
  object["mapper_tile_cols"] = shape.mapper_cols;
  return object;
}

// Serializes one candidate while preserving task order. The candidate ID is a
// sequential index, so no task-body identity or file-derived metadata is
// required to interpret it within its validated manifest.
llvm::json::Object candidateJson(const Candidate &candidate) {
  llvm::json::Array choices;
  for (const TaskShapeChoice &choice : candidate.choices) {
    llvm::json::Object record;
    record["task"] = choice.task;
    record["trip_count"] = choice.trip_count;
    record["shape"] = shapeJson(choice.shape);
    choices.push_back(std::move(record));
  }
  llvm::json::Object record;
  record["record_type"] = "candidate";
  record["schema"] = kCandidateSchema.str();
  record["candidate_id"] = candidate.id;
  record["task_shapes"] = std::move(choices);
  return record;
}

} // namespace analytical_candidates
} // namespace neura
} // namespace amoeba
} // namespace mlir

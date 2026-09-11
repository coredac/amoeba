//===- SpatialTaskCandidateSpace.cpp -----------------------------------===//
//
// Implements construction and traversal of the static rectangular
// task-shape candidate space.
//
//===----------------------------------------------------------------------===//

#include "SpatialTaskCandidateSpace.h"

#include "Backend/Neura/Orchestration/AnalyticalTaskDSE/SpatialDSEOrchestration.h"

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
namespace analytical_dse {

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
// only needs to place already selected tasks and should not own DSE packing.
static std::optional<SmallVector<std::pair<int, int>>>
getShapeCells(const CgraShape &shape, int gridRows, int gridCols) {
  if (shape.rows <= 0 || shape.cols <= 0 || shape.rows > gridRows ||
      shape.cols > gridCols)
    return std::nullopt;

  SmallVector<std::pair<int, int>> cells;
  if (shape.is_rectangular) {
    for (int row = 0; row < shape.rows; ++row)
      for (int col = 0; col < shape.cols; ++col)
        cells.push_back({col, row});
    return cells;
  }
  if (shape.cgra_positions.empty())
    return std::nullopt;

  DenseSet<int64_t> seen;
  for (auto [col, row] : shape.cgra_positions) {
    if (col < 0 || col >= shape.cols || row < 0 || row >= shape.rows)
      return std::nullopt;
    int64_t key = static_cast<int64_t>(row) * shape.cols + col;
    if (!seen.insert(key).second)
      return std::nullopt;
    cells.push_back({col, row});
  }
  return cells;
}

static bool placeShapeChoices(size_t taskIndex,
                              ArrayRef<SmallVector<CgraShape>> shapeChoices,
                              int gridRows, int gridCols,
                              MutableArrayRef<uint8_t> occupied) {
  if (taskIndex == shapeChoices.size())
    return true;

  for (const CgraShape &shape : shapeChoices[taskIndex]) {
    std::optional<SmallVector<std::pair<int, int>>> cells =
        getShapeCells(shape, gridRows, gridCols);
    if (!cells)
      continue;
    for (int originRow = 0; originRow + shape.rows <= gridRows; ++originRow) {
      for (int originCol = 0; originCol + shape.cols <= gridCols; ++originCol) {
        bool overlaps = llvm::any_of(*cells, [&](auto offset) {
          auto [col, row] = offset;
          return occupied[static_cast<size_t>((originRow + row) * gridCols +
                                              originCol + col)] != 0;
        });
        if (overlaps)
          continue;

        for (auto [col, row] : *cells)
          occupied[static_cast<size_t>((originRow + row) * gridCols +
                                       originCol + col)] = 1;
        if (placeShapeChoices(taskIndex + 1, shapeChoices, gridRows, gridCols,
                              occupied))
          return true;
        for (auto [col, row] : *cells)
          occupied[static_cast<size_t>((originRow + row) * gridCols +
                                       originCol + col)] = 0;
      }
    }
  }
  return false;
}

// Checks exact simultaneous placement for fixed-orientation spatial shapes.
static bool canShapesFitOnGrid(ArrayRef<CgraShape> taskShapes, int gridRows,
                               int gridCols) {
  if (gridRows <= 0 || gridCols <= 0 ||
      static_cast<uint64_t>(gridRows) * static_cast<uint64_t>(gridCols) >
          std::numeric_limits<size_t>::max())
    return false;

  const uint64_t gridArea = static_cast<uint64_t>(gridRows) * gridCols;
  uint64_t occupiedCells = 0;
  for (const CgraShape &shape : taskShapes) {
    std::optional<SmallVector<std::pair<int, int>>> cells =
        getShapeCells(shape, gridRows, gridCols);
    if (!cells || cells->size() > gridArea - occupiedCells)
      return false;
    occupiedCells += cells->size();
  }

  SmallVector<CgraShape> largestFirst(taskShapes.begin(), taskShapes.end());
  llvm::sort(largestFirst, [](const CgraShape &lhs, const CgraShape &rhs) {
    auto cellCount = [](const CgraShape &shape) {
      return shape.is_rectangular
                 ? static_cast<int64_t>(shape.rows) * shape.cols
                 : static_cast<int64_t>(shape.cgra_positions.size());
    };
    return cellCount(lhs) > cellCount(rhs);
  });
  SmallVector<SmallVector<CgraShape>> shapeChoices;
  shapeChoices.reserve(largestFirst.size());
  for (const CgraShape &shape : largestFirst)
    shapeChoices.push_back({shape});

  SmallVector<uint8_t> occupied(static_cast<size_t>(gridRows) * gridCols, 0);
  return placeShapeChoices(0, shapeChoices, gridRows, gridCols, occupied);
}

// Enumerates every legal static physical rectangle and derives its mapper
// dimensions from the architecture getters. The deterministic order defines
// the mixed-radix alphabet for candidate IDs.
// TODO: Extend the analytical candidate schema and its consumers to represent
// non-rectangular shapes. The analytical search intentionally enumerates only
// fixed-orientation rectangles until that contract exists end to end.
SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                 int64_t gridCols,
                                                 int64_t perCgraRows,
                                                 int64_t perCgraCols,
                                                 int64_t maxCgrasPerTask) {
  // The candidate domain contains only concrete integer rectangles.
  SmallVector<RectShape> result;
  if (gridRows <= 0 || gridCols <= 0 || perCgraRows <= 0 || perCgraCols <= 0 ||
      maxCgrasPerTask <= 0)
    return result;

  const int64_t gridSize =
      gridRows > std::numeric_limits<int64_t>::max() / gridCols
          ? std::numeric_limits<int64_t>::max()
          : gridRows * gridCols;
  for (int64_t count = 1; count <= std::min(gridSize, maxCgrasPerTask);
       ++count) {
    for (const CgraShape &physicalShape :
         taskflow::SpatialDSEOrchestration::getRectangularShapes(
             static_cast<int>(count), static_cast<int>(gridRows),
             static_cast<int>(gridCols))) {
      if (physicalShape.rows >
              std::numeric_limits<int64_t>::max() / perCgraRows ||
          physicalShape.cols >
              std::numeric_limits<int64_t>::max() / perCgraCols)
        continue;
      int64_t mapperRows = physicalShape.rows * perCgraRows;
      int64_t mapperCols = physicalShape.cols * perCgraCols;
      result.push_back(
          {physicalShape.rows, physicalShape.cols, mapperRows, mapperCols});
    }
  }
  return result;
}

int64_t operationCappedMaximumPhysicalCgras(int64_t materializedOperationCount,
                                            int64_t perCgraRows,
                                            int64_t perCgraCols,
                                            int64_t maxCgrasPerTask) {
  if (perCgraRows <= 0 || perCgraCols <= 0 || maxCgrasPerTask <= 0 ||
      perCgraRows > std::numeric_limits<int64_t>::max() / perCgraCols)
    return 0;
  int64_t tilesPerCgra = perCgraRows * perCgraCols;
  int64_t count = std::max<int64_t>(1, materializedOperationCount);
  int64_t cap = 1 + (count - 1) / tilesPerCgra;
  int64_t halfFull = 1 + (tilesPerCgra - 1) / 2;
  if (count >= halfFull)
    cap = std::max<int64_t>(2, cap);
  return std::min(maxCgrasPerTask, cap);
}

SmallVector<SmallVector<RectShape>> buildOperationCappedShapeAlphabets(
    ArrayRef<TaskFact> tasks, ArrayRef<RectShape> shapes, int64_t perCgraRows,
    int64_t perCgraCols, int64_t maxCgrasPerTask) {
  SmallVector<SmallVector<RectShape>> result(tasks.size());
  for (auto [taskIndex, task] : llvm::enumerate(tasks)) {
    int64_t cap = operationCappedMaximumPhysicalCgras(
        task.materializedOperationCount, perCgraRows, perCgraCols,
        maxCgrasPerTask);
    for (const RectShape &shape : shapes)
      if (shape.cgraCount() <= cap)
        result[taskIndex].push_back(shape);
  }
  return result;
}

bool ConcurrentPackingCache::canPack(ArrayRef<RectShape> shapes) {
  Key key;
  key.reserve(shapes.size());
  for (const RectShape &shape : shapes)
    key.push_back({shape.rows, shape.cols});
  llvm::sort(key, [](const auto &lhs, const auto &rhs) {
    const int64_t lhsArea = lhs.first * lhs.second;
    const int64_t rhsArea = rhs.first * rhs.second;
    if (lhsArea != rhsArea)
      return lhsArea > rhsArea;
    if (std::max(lhs.first, lhs.second) != std::max(rhs.first, rhs.second))
      return std::max(lhs.first, lhs.second) > std::max(rhs.first, rhs.second);
    return lhs > rhs;
  });

  auto found = results_.find(key);
  if (found != results_.end())
    return found->second;

  bool result = false;
  if (gridRows_ > 0 && gridCols_ > 0 &&
      gridRows_ <= std::numeric_limits<int>::max() &&
      gridCols_ <= std::numeric_limits<int>::max()) {
    SmallVector<CgraShape> fixedShapes;
    fixedShapes.reserve(shapes.size());
    bool dimensionsFit = true;
    for (const RectShape &shape : shapes) {
      if (shape.rows <= 0 || shape.cols <= 0 ||
          shape.rows > std::numeric_limits<int>::max() ||
          shape.cols > std::numeric_limits<int>::max()) {
        dimensionsFit = false;
        break;
      }
      fixedShapes.push_back({static_cast<int>(shape.rows),
                             static_cast<int>(shape.cols),
                             true,
                             {}});
    }
    if (dimensionsFit)
      result = canShapesFitOnGrid(fixedShapes, static_cast<int>(gridRows_),
                                  static_cast<int>(gridCols_));
  }
  results_.emplace(std::move(key), result);
  return result;
}

bool visitConcurrentlyPackableShapeTuples(
    ArrayRef<SmallVector<RectShape>> shapesByTask,
    ConcurrentPackingCache &packing, ShapeIndexTupleConsumer consume) {
  const int64_t gridRows = packing.gridRows();
  const int64_t gridCols = packing.gridCols();
  if (shapesByTask.empty() || gridRows <= 0 || gridCols <= 0 ||
      llvm::any_of(shapesByTask,
                   [](const auto &shapes) { return shapes.empty(); }) ||
      gridRows > std::numeric_limits<int64_t>::max() / gridCols)
    return true;
  const int64_t gridArea = gridRows * gridCols;
  // Every supported task consumes at least one physical CGRA. This prevents a
  // large task list from expanding an obviously empty Cartesian product.
  if (shapesByTask.size() > static_cast<size_t>(gridArea))
    return true;

  SmallVector<size_t> selectedIndices;
  SmallVector<RectShape> selectedShapes;
  uint64_t validIndex = 0;
  std::function<bool(size_t, int64_t)> visit = [&](size_t taskIndex,
                                                   int64_t selectedArea) {
    if (taskIndex == shapesByTask.size()) {
      if (!packing.canPack(selectedShapes))
        return true;
      if (!consume(validIndex, selectedIndices))
        return false;
      ++validIndex;
      return true;
    }

    // Shapes remain in their declared deterministic order. The area test
    // removes only tuples that cannot possibly fit; geometry is checked
    // exactly after one shape has been chosen for every task.
    for (auto [shapeIndex, shape] : llvm::enumerate(shapesByTask[taskIndex])) {
      const int64_t area = shape.cgraCount();
      if (area <= 0 || area > gridArea - selectedArea)
        continue;
      selectedIndices.push_back(shapeIndex);
      selectedShapes.push_back(shape);
      if (!visit(taskIndex + 1, selectedArea + area))
        return false;
      selectedShapes.pop_back();
      selectedIndices.pop_back();
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
  object["mapper_tile_rows"] = shape.mapperRows;
  object["mapper_tile_cols"] = shape.mapperCols;
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
    record["trip_count"] = choice.tripCount;
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

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

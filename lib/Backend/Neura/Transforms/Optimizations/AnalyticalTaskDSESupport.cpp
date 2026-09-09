//===- AnalyticalTaskDSESupport.cpp - Shared task DSE support ------------===//
//
// Implements the records and file protocol shared by the static analytical
// task-DSE passes.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskDSESupport.h"

#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <system_error>
#include <utility>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

// Returns a lowercase SHA-256. File hashes use raw bytes so any input change
// invalidates a manifest produced from those bytes.
static std::string sha256(StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

// Fingerprints the exact YAML selected by --architecture-spec. Dimensions
// alone are insufficient because two same-sized machines can have different
// FU, memory, latency, or routing capabilities.
FailureOr<std::string> currentArchitectureSha256(std::string &error) {
  StringRef path = mlir::amoeba::getNeuraArchitectureSpecFile();
  if (path.empty()) {
    error = "analytical task DSE requires --architecture-spec";
    return failure();
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot fingerprint architecture specification " + path.str() +
            ": " + buffer.getError().message();
    return failure();
  }
  return sha256((*buffer)->getBuffer());
}

// Converts the physical CGRA rectangle into the string stored in the
// Taskflow `cgra_shape` attribute, such as `1x2`.
std::string RectShape::toCgraShapeAttrValue() const {
  return std::to_string(rows) + "x" + std::to_string(cols);
}

// Resolves the positive static trip count stored with each task. An explicit
// `trip_count` is authoritative; otherwise, constant Taskflow counter bounds
// supply the count. A task without a counter represents one execution.
static FailureOr<int64_t> resolveAnalyticalTripCount(TaskflowTaskOp task,
                                                     std::string &error) {
  if (auto attr = task->getAttrOfType<IntegerAttr>("trip_count")) {
    if (attr.getInt() <= 0) {
      error =
          "task " + task.getTaskName().str() + " has non-positive trip_count";
      return failure();
    }
    return attr.getInt();
  }
  FailureOr<std::optional<int64_t>> inferred =
      inferStaticTaskTripCount(task, error);
  if (failed(inferred)) {
    error += "; static shape selection requires constant counter bounds";
    return failure();
  }
  return inferred->value_or(1);
}

// Produces a stable identity for the current task computation. We deliberately
// remove DSE outputs and measurements so materializing a shape does not make
// an otherwise identical task look new. A body edit, however, changes this
// hash and invalidates an old candidate manifest.
static std::string taskBodySha256(TaskflowTaskOp task) {
  Operation *clone = task->clone();
  auto destroyClone = llvm::make_scope_exit([&] { clone->destroy(); });
  clone->setAttr("task_name",
                 StringAttr::get(task.getContext(), "__analytical_task__"));
  for (StringRef attribute :
       {"trip_count", "cgra_count", "cgra_shape", "compiled_ii", "profile_info",
        "task_orchestration_info", "replicas", "tiling", "est_latency"})
    clone->removeAttr(attribute);
  clone->removeAttr("amoeba.analytical_shape_orientation_fixed");
  // This attribute is an exported copy of the hash being computed. Excluding
  // it keeps the identity stable when an already-bound IR is enumerated again.
  clone->removeAttr(kSourceTaskBodyShaAttr);
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  OpPrintingFlags flags;
  flags.printGenericOpForm().useLocalScope();
  clone->print(stream, flags);
  stream.flush();
  return sha256(printed);
}

// Collects task names, operations, and static trip counts in walk order. The
// order is the task axis used by shape-tuple enumeration, so duplicate names
// are rejected before they can make candidate records ambiguous.
FailureOr<SmallVector<TaskFact>>
collectAnalyticalTaskFacts(func::FuncOp func, std::string &error) {
  SmallVector<TaskFact> tasks;
  llvm::StringSet<> names;
  WalkResult walkResult = func.walk([&](TaskflowTaskOp task) {
    std::string name = task.getTaskName().str();
    if (!names.insert(name).second) {
      error = "duplicate task name " + name;
      return WalkResult::interrupt();
    }
    FailureOr<int64_t> tripCount = resolveAnalyticalTripCount(task, error);
    if (failed(tripCount))
      return WalkResult::interrupt();
    tasks.push_back({task, std::move(name), taskBodySha256(task), *tripCount});
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  if (tasks.empty()) {
    error = "function contains no taskflow.task operations";
    return failure();
  }
  return tasks;
}

// Enumerates every legal static physical rectangle and derives its mapper
// dimensions from the architecture getters. The deterministic order defines
// the mixed-radix alphabet for candidate IDs.
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
  const int64_t maxCount = std::min(maxCgrasPerTask, gridSize);
  for (int64_t count = 1; count <= maxCount; ++count) {
    for (const CgraShape &physicalShape : taskflow::getRectangularShapes(
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

// Tries to place the fixed-orientation rectangles exactly. This is a small
// backtracking search over physical CGRA cells, not a solver and not the
// downstream placement heuristic. Sorting large rectangles first only changes
// search speed; it does not remove any legal placement.
static bool placeRectangles(size_t rectangleIndex,
                            ArrayRef<RectShape> rectangles, int64_t gridRows,
                            int64_t gridCols,
                            MutableArrayRef<uint8_t> occupied) {
  if (rectangleIndex == rectangles.size())
    return true;

  const RectShape &shape = rectangles[rectangleIndex];
  for (int64_t originRow = 0; originRow + shape.rows <= gridRows; ++originRow) {
    for (int64_t originCol = 0; originCol + shape.cols <= gridCols;
         ++originCol) {
      bool overlaps = false;
      for (int64_t row = 0; row < shape.rows && !overlaps; ++row) {
        for (int64_t col = 0; col < shape.cols; ++col) {
          size_t cell = static_cast<size_t>((originRow + row) * gridCols +
                                            originCol + col);
          if (occupied[cell]) {
            overlaps = true;
            break;
          }
        }
      }
      if (overlaps)
        continue;

      for (int64_t row = 0; row < shape.rows; ++row)
        for (int64_t col = 0; col < shape.cols; ++col)
          occupied[static_cast<size_t>((originRow + row) * gridCols +
                                       originCol + col)] = 1;
      if (placeRectangles(rectangleIndex + 1, rectangles, gridRows, gridCols,
                          occupied))
        return true;
      for (int64_t row = 0; row < shape.rows; ++row)
        for (int64_t col = 0; col < shape.cols; ++col)
          occupied[static_cast<size_t>((originRow + row) * gridCols +
                                       originCol + col)] = 0;
    }
  }
  return false;
}

// The static shape-selection contract requires all task rectangles to be
// resident simultaneously. The area check is only a cheap necessary condition;
// backtracking is required because fixed rectangle orientations can conflict
// even when their total area fits the grid.
static bool canPackSimultaneously(ArrayRef<RectShape> selected,
                                  int64_t gridRows, int64_t gridCols) {
  if (gridRows <= 0 || gridCols <= 0 ||
      gridRows > std::numeric_limits<int64_t>::max() / gridCols)
    return false;
  const int64_t gridArea = gridRows * gridCols;
  int64_t selectedArea = 0;
  SmallVector<RectShape> largestFirst(selected.begin(), selected.end());
  for (const RectShape &shape : largestFirst) {
    if (shape.rows <= 0 || shape.cols <= 0 || shape.rows > gridRows ||
        shape.cols > gridCols || shape.rows > gridArea / shape.cols ||
        selectedArea > gridArea - shape.cgraCount())
      return false;
    selectedArea += shape.cgraCount();
  }
  llvm::sort(largestFirst, [](const RectShape &lhs, const RectShape &rhs) {
    if (lhs.cgraCount() != rhs.cgraCount())
      return lhs.cgraCount() > rhs.cgraCount();
    if (std::max(lhs.rows, lhs.cols) != std::max(rhs.rows, rhs.cols))
      return std::max(lhs.rows, lhs.cols) > std::max(rhs.rows, rhs.cols);
    return std::tie(lhs.rows, lhs.cols) > std::tie(rhs.rows, rhs.cols);
  });

  SmallVector<uint8_t> occupied(static_cast<size_t>(gridArea), 0);
  return placeRectangles(0, largestFirst, gridRows, gridCols, occupied);
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
  bool result = canPackSimultaneously(shapes, gridRows_, gridCols_);
  results_.emplace(std::move(key), result);
  return result;
}

bool visitConcurrentlyPackableShapeTuples(size_t taskCount,
                                          ArrayRef<RectShape> shapes,
                                          ConcurrentPackingCache &packing,
                                          ShapeIndexTupleConsumer consume) {
  const int64_t gridRows = packing.gridRows();
  const int64_t gridCols = packing.gridCols();
  if (taskCount == 0 || shapes.empty() || gridRows <= 0 || gridCols <= 0 ||
      gridRows > std::numeric_limits<int64_t>::max() / gridCols)
    return true;
  const int64_t gridArea = gridRows * gridCols;
  // Every supported task consumes at least one physical CGRA. This prevents a
  // large task list from expanding an obviously empty Cartesian product.
  if (taskCount > static_cast<size_t>(gridArea))
    return true;

  SmallVector<size_t> selectedIndices;
  SmallVector<RectShape> selectedShapes;
  uint64_t validIndex = 0;
  std::function<bool(size_t, int64_t)> visit = [&](size_t taskIndex,
                                                   int64_t selectedArea) {
    if (taskIndex == taskCount) {
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
    for (auto [shapeIndex, shape] : llvm::enumerate(shapes)) {
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

// Formats the deterministic candidate ID assigned by enumeration. For
// example, the first candidate is `candidate-0`, and the next is
// `candidate-1`. The numeric suffix indexes only concurrently packable tuples,
// so IDs stay contiguous after impossible shape assignments are removed.
std::string makeSequentialCandidateId(uint64_t index) {
  return "candidate-" + std::to_string(index);
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

// Writes one JSON object as a single JSONL record.
void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object) {
  os << llvm::json::Value(std::move(object)) << "\n";
}

// Publishes a complete output atomically so consumers never read a partial
// candidate manifest.
bool writeAtomically(StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error) {
  if (output.empty()) {
    error = "output path is required";
    return false;
  }
  SmallString<256> pattern(output);
  pattern += ".tmp-%%%%%%";
  SmallString<256> temporary;
  int descriptor = -1;
  std::error_code ec =
      llvm::sys::fs::createUniqueFile(pattern, descriptor, temporary);
  if (ec) {
    error = "cannot create temporary output " + temporary.str().str() + ": " +
            ec.message();
    return false;
  }

  bool ok = false;
  {
    llvm::raw_fd_ostream os(descriptor, /*shouldClose=*/true);
    ok = writeBody(os);
    os.flush();
    if (os.has_error()) {
      error = "failed while writing " + output.str();
      ok = false;
    }
  }
  if (!ok) {
    llvm::sys::fs::remove(temporary);
    return false;
  }
  ec = llvm::sys::fs::rename(temporary, output);
  if (ec) {
    error = "cannot publish " + output.str() + ": " + ec.message();
    llvm::sys::fs::remove(temporary);
    return false;
  }
  return true;
}

// Selects the requested Taskflow function, or infers it when exactly one
// function contains tasks. This keeps every file-producing pass consistent.
FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module, StringRef requested,
                                           std::string &error) {
  if (!requested.empty()) {
    auto function = module.lookupSymbol<func::FuncOp>(requested);
    if (!function) {
      error = "requested function " + requested.str() + " does not exist";
      return failure();
    }
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (!hasTask) {
      error = "requested function " + requested.str() +
              " contains no taskflow.task operations";
      return failure();
    }
    return function;
  }

  SmallVector<func::FuncOp> taskFunctions;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (hasTask)
      taskFunctions.push_back(function);
  }
  if (taskFunctions.size() != 1) {
    error = "expected exactly one function containing Taskflow tasks; use the "
            "function option when the module contains more than one";
    return failure();
  }
  return taskFunctions.front();
}

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

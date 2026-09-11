//===- AnalyticalTaskCandidateSpace.h -------------------------*- C++ -*-===//
//
// Defines the facts, shapes, and traversal used to construct the finite
// analytical task-shape candidate space. It also owns canonical IDs and JSON
// serialization helpers, but no parsing, scoring, materialization, or mapping.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_TASK_CANDIDATE_SPACE_H
#define AMOEBA_ANALYTICAL_TASK_CANDIDATE_SPACE_H

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

inline constexpr llvm::StringLiteral kCandidateSchema =
    "amoeba-analytical-task-candidates";
inline constexpr llvm::StringLiteral kShapePolicy =
    "static-oriented-rectangles";
inline constexpr llvm::StringLiteral kSearchScope =
    "static-shape-concurrent-fit";
inline constexpr llvm::StringLiteral kSpatialCapacityPolicy =
    "all-tasks-simultaneous-exact-pack";
inline constexpr llvm::StringLiteral kSymbolDynamicTripCountKind =
    "symbol_dynamic";
inline constexpr llvm::StringLiteral kSourceTaskBodyShaAttr =
    "amoeba.source_task_body_sha256";

// Stores one physical-CGRA rectangle and its corresponding mapper dimensions.
struct RectShape {
  int64_t rows = 1;
  int64_t cols = 1;
  int64_t mapperRows = 1;
  int64_t mapperCols = 1;

  int64_t cgraCount() const { return rows * cols; }
  std::string toCgraShapeAttrValue() const;
};

// Stores immutable identity and any compile-time trip count from one Taskflow
// task. A missing trip count denotes a symbol-bound counter chain whose value
// is fixed before the task launches but is not known at compile time.
struct TaskFact {
  taskflow::TaskflowTaskOp op;
  std::string name;
  std::string bodySha256;
  std::optional<int64_t> tripCount = int64_t{1};
};

// Stores one task's shape choice within a program candidate.
struct TaskShapeChoice {
  std::string task;
  std::optional<int64_t> tripCount = int64_t{1};
  RectShape shape;
};

// Stores one ordered shape choice for every Taskflow task.
struct Candidate {
  std::string id;
  llvm::SmallVector<TaskShapeChoice> choices;
};

using ShapeIndexTupleConsumer =
    llvm::function_ref<bool(uint64_t, llvm::ArrayRef<size_t>)>;

// Memoizes exact physical-grid packing by the sorted multiset of oriented
// rectangles. Task names and task order do not affect whether the rectangles
// fit, so many ordered candidates share one small backtracking result.
class ConcurrentPackingCache {
public:
  ConcurrentPackingCache(int64_t gridRows, int64_t gridCols)
      : gridRows_(gridRows), gridCols_(gridCols) {}

  bool canPack(llvm::ArrayRef<RectShape> shapes);
  int64_t gridRows() const { return gridRows_; }
  int64_t gridCols() const { return gridCols_; }

private:
  using Key = std::vector<std::pair<int64_t, int64_t>>;

  int64_t gridRows_ = 0;
  int64_t gridCols_ = 0;
  std::map<Key, bool> results_;
};

FailureOr<llvm::SmallVector<TaskFact>>
collectAnalyticalTaskFacts(func::FuncOp func, std::string &error);
llvm::SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                       int64_t gridCols,
                                                       int64_t perCgraRows,
                                                       int64_t perCgraCols);
// Visits every shape tuple that admits a simultaneous, non-overlapping
// placement on the physical grid. `shapeIndices` follows task order and indexes
// `shapes`; the valid candidate index is contiguous and starts at zero. Returns
// false only when the consumer requests an early stop.
bool visitConcurrentlyPackableShapeTuples(size_t taskCount,
                                          llvm::ArrayRef<RectShape> shapes,
                                          ConcurrentPackingCache &packing,
                                          ShapeIndexTupleConsumer consume);
std::string makeSequentialCandidateId(uint64_t index);
llvm::json::Object candidateJson(const Candidate &candidate);
void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object);
bool writeAtomically(llvm::StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error);
FailureOr<std::string> currentArchitectureSha256(std::string &error);
FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module,
                                           llvm::StringRef requested,
                                           std::string &error);
} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_TASK_CANDIDATE_SPACE_H

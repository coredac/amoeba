//===- SpatialTaskCandidateSpace.h -----------------------------*- C++ -*-===//
//
// Defines the fixed-orientation spatial shape alphabet and its exact
// simultaneous packing traversal. A future temporal candidate space can live
// beside this file while sharing task facts and output helpers.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H
#define AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H

#include "AnalyticalTaskCandidateCommon.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/JSON.h"

#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

inline constexpr llvm::StringLiteral kShapePolicy =
    "static-oriented-rectangles";
inline constexpr llvm::StringLiteral kSearchScope =
    "static-shape-concurrent-fit";
inline constexpr llvm::StringLiteral kSpatialCapacityPolicy =
    "all-tasks-simultaneous-exact-pack";
inline constexpr llvm::StringLiteral kShapePruningPolicy =
    "none";

// Stores one physical-CGRA rectangle and its corresponding mapper dimensions.
struct RectShape {
  int64_t rows = 1;
  int64_t cols = 1;
  int64_t mapperRows = 1;
  int64_t mapperCols = 1;

  int64_t cgraCount() const { return rows * cols; }
  std::string toCgraShapeAttrValue() const;
};

// Stores one task's shape choice within a program candidate.
struct TaskShapeChoice {
  std::string task;
  int64_t tripCount = 1;
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

llvm::SmallVector<RectShape> enumerateStaticRectShapes(int64_t gridRows,
                                                       int64_t gridCols,
                                                       int64_t perCgraRows,
                                                       int64_t perCgraCols,
                                                       int64_t maxCgrasPerTask);
// Visits every shape tuple that admits a simultaneous, non-overlapping
// placement on the physical grid. `shapeIndices` follows task order and indexes
// the corresponding task's shape alphabet; the valid candidate index is
// contiguous and starts at zero. Returns
// false only when the consumer requests an early stop.
bool visitConcurrentlyPackableShapeTuples(
    llvm::ArrayRef<llvm::SmallVector<RectShape>> shapesByTask,
    ConcurrentPackingCache &packing, ShapeIndexTupleConsumer consume);
llvm::json::Object candidateJson(const Candidate &candidate);
} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_SPATIAL_TASK_CANDIDATE_SPACE_H

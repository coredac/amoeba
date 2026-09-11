// Spatial candidate shape and counter facts for analytical task DSE.

#include "Backend/Neura/Orchestration/AnalyticalTaskDSE/SpatialShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <functional>
#include <limits>

using llvm::SmallVector;

namespace mlir {
namespace taskflow {
namespace static_shape {

SmallVector<CgraShape> getRectangularShapes(int cgra_count, int grid_rows,
                                            int grid_cols) {
  SmallVector<CgraShape> shapes;
  if (cgra_count <= 0 || grid_rows <= 0 || grid_cols <= 0)
    return shapes;

  for (int rows = 1; rows <= grid_rows; ++rows) {
    if (cgra_count % rows != 0)
      continue;
    int cols = cgra_count / rows;
    if (cols <= grid_cols)
      shapes.push_back({rows, cols, true, {}});
  }
  return shapes;
}

FailureOr<std::optional<int64_t>> inferStaticTaskTripCount(TaskflowTaskOp task,
                                                           std::string &error) {
  SmallVector<TaskflowCounterOp> counters;
  task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
  if (counters.empty())
    return std::optional<int64_t>{};

  if (!task.getBody().hasOneBlock()) {
    error = "task " + task.getTaskName().str() +
            " must contain exactly one block to infer a static trip count";
    return failure();
  }

  SmallVector<TaskflowCounterOp> roots;
  llvm::DenseMap<Value, SmallVector<TaskflowCounterOp>> children;
  for (TaskflowCounterOp counter : counters) {
    if (Value parent = counter.getParentIndex())
      children[parent].push_back(counter);
    else
      roots.push_back(counter);
  }
  if (roots.empty()) {
    error = "task " + task.getTaskName().str() +
            " has counters but no root counter";
    return failure();
  }

  auto constantIndex = [](Value value) -> FailureOr<int64_t> {
    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>())
      return constant.value();
    return failure();
  };
  auto counterTripCount = [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    FailureOr<int64_t> lower = constantIndex(counter.getLowerBound());
    FailureOr<int64_t> upper = constantIndex(counter.getUpperBound());
    FailureOr<int64_t> step = constantIndex(counter.getStep());
    if (failed(lower) || failed(upper) || failed(step))
      return failure();
    if (*step <= 0 || *upper <= *lower ||
        (*lower < 0 && *upper > std::numeric_limits<int64_t>::max() + *lower))
      return failure();

    int64_t distance = *upper - *lower;
    return 1 + (distance - 1) / *step;
  };

  llvm::DenseSet<Operation *> active;
  llvm::DenseSet<Operation *> visited;
  std::function<FailureOr<int64_t>(TaskflowCounterOp)> chainTripCount =
      [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    Operation *operation = counter.getOperation();
    if (!active.insert(operation).second || visited.contains(operation)) {
      error = "task " + task.getTaskName().str() +
              " has a cyclic or multiply referenced counter chain";
      return failure();
    }

    FailureOr<int64_t> count = counterTripCount(counter);
    if (failed(count)) {
      error = "task " + task.getTaskName().str() +
              " requires constant counter bounds, a positive step, a "
              "non-empty range, and a trip count within int64";
      return failure();
    }

    int64_t longest_child_chain = 1;
    auto found = children.find(counter.getCounterIndex());
    if (found != children.end()) {
      for (TaskflowCounterOp child : found->second) {
        FailureOr<int64_t> child_count = chainTripCount(child);
        if (failed(child_count))
          return failure();
        longest_child_chain = std::max(longest_child_chain, *child_count);
      }
    }
    if (*count > std::numeric_limits<int64_t>::max() / longest_child_chain) {
      error = "task " + task.getTaskName().str() +
              " requires constant counter bounds, a positive step, a "
              "non-empty range, and a trip count within int64";
      return failure();
    }

    active.erase(operation);
    visited.insert(operation);
    return *count * longest_child_chain;
  };

  int64_t total = 1;
  for (TaskflowCounterOp root : roots) {
    FailureOr<int64_t> root_count = chainTripCount(root);
    if (failed(root_count))
      return failure();
    total = std::max(total, *root_count);
  }
  if (visited.size() != counters.size()) {
    error = "task " + task.getTaskName().str() +
            " has a counter disconnected from every root";
    return failure();
  }
  return std::optional<int64_t>{total};
}

} // namespace static_shape
} // namespace taskflow
} // namespace mlir

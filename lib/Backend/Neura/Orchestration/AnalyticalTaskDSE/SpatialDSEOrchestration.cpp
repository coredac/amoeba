// Orchestrates a materialized spatial analytical DSE candidate.

#include "Backend/Neura/Orchestration/AnalyticalTaskDSE/SpatialDSEOrchestration.h"

#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <string>

using llvm::SmallVector;

namespace mlir {
namespace taskflow {

SmallVector<CgraShape>
SpatialDSEOrchestration::getRectangularShapes(int cgraCount, int gridRows,
                                              int gridCols) {
  SmallVector<CgraShape> shapes;
  if (cgraCount <= 0 || gridRows <= 0 || gridCols <= 0)
    return shapes;

  for (int rows = 1; rows <= gridRows; ++rows) {
    if (cgraCount % rows != 0)
      continue;
    int cols = cgraCount / rows;
    if (cols <= gridCols)
      shapes.push_back({rows, cols, true, {}});
  }
  return shapes;
}

FailureOr<std::optional<int64_t>>
SpatialDSEOrchestration::inferStaticTaskTripCount(TaskflowTaskOp task,
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

    int64_t longestChildChain = 1;
    auto found = children.find(counter.getCounterIndex());
    if (found != children.end()) {
      for (TaskflowCounterOp child : found->second) {
        FailureOr<int64_t> childCount = chainTripCount(child);
        if (failed(childCount))
          return failure();
        longestChildChain = std::max(longestChildChain, *childCount);
      }
    }
    if (*count > std::numeric_limits<int64_t>::max() / longestChildChain) {
      error = "task " + task.getTaskName().str() +
              " requires constant counter bounds, a positive step, a "
              "non-empty range, and a trip count within int64";
      return failure();
    }

    active.erase(operation);
    visited.insert(operation);
    return *count * longestChildChain;
  };

  int64_t total = 1;
  for (TaskflowCounterOp root : roots) {
    FailureOr<int64_t> rootCount = chainTripCount(root);
    if (failed(rootCount))
      return failure();
    total = std::max(total, *rootCount);
  }
  if (visited.size() != counters.size()) {
    error = "task " + task.getTaskName().str() +
            " has a counter disconnected from every root";
    return failure();
  }
  return std::optional<int64_t>{total};
}

namespace {

// Parses and validates the compact cgra_shape spelling produced by the
// analytical candidate materializer.  Keeping this check next to the static
// strategy makes malformed or incomplete materialization fail before the
// scheduler can consume a partial resource decision.
bool validateShapeString(StringRef shape_text, int cgra_count, int grid_rows,
                         int grid_cols, std::string &error) {
  size_t x_pos = shape_text.find('x');
  if (x_pos == StringRef::npos || x_pos == 0 ||
      x_pos + 1 >= shape_text.size() ||
      shape_text.find('x', x_pos + 1) != StringRef::npos) {
    error = "cgra_shape must be a fixed rectangle of the form 'rowsxcols'";
    return false;
  }

  int rows = 0;
  int cols = 0;
  if (shape_text.take_front(x_pos).getAsInteger(10, rows) || rows <= 0 ||
      shape_text.drop_front(x_pos + 1).getAsInteger(10, cols) || cols <= 0) {
    error = "cgra_shape has invalid rectangle dimensions";
    return false;
  }
  if (rows > grid_rows || cols > grid_cols) {
    error = "cgra_shape rectangle does not fit the CGRA grid";
    return false;
  }
  if (static_cast<int64_t>(rows) * cols != cgra_count) {
    error = "cgra_shape area does not equal cgra_count";
    return false;
  }
  return true;
}
} // namespace

bool SpatialDSEOrchestration::validateFixedShapeAttributes(
    func::FuncOp func) const {
  auto candidate_id =
      func->getAttrOfType<StringAttr>("analytical_task_candidate_id");
  if (!candidate_id || candidate_id.getValue().empty()) {
    func.emitError()
        << "analytical DSE spatial orchestration requires a non-empty "
           "analytical_task_candidate_id attribute";
    return false;
  }

  bool valid = true;
  func.walk([&](TaskflowTaskOp task) {
    if (!valid) {
      return;
    }
    if (!task->getAttrOfType<UnitAttr>(
            "amoeba.analytical_shape_orientation_fixed")) {
      task.emitError() << "analytical DSE spatial orchestration requires "
                          "amoeba.analytical_shape_orientation_fixed";
      valid = false;
      return;
    }

    auto cgra_count = task->getAttrOfType<IntegerAttr>("cgra_count");
    if (!cgra_count || cgra_count.getInt() <= 0 ||
        cgra_count.getInt() > std::numeric_limits<int>::max()) {
      task.emitError() << "analytical DSE spatial orchestration requires a "
                          "positive cgra_count";
      valid = false;
      return;
    }

    auto cgra_shape = task->getAttrOfType<StringAttr>("cgra_shape");
    if (!cgra_shape || cgra_shape.getValue().empty()) {
      task.emitError() << "analytical DSE spatial orchestration requires a "
                          "non-empty cgra_shape";
      valid = false;
      return;
    }

    std::string error;
    if (!validateShapeString(cgra_shape.getValue(),
                             static_cast<int>(cgra_count.getInt()), grid_rows_,
                             grid_cols_, error)) {
      task.emitError() << "invalid fixed cgra_shape: " << error;
      valid = false;
    }
  });
  return valid;
}

void SpatialDSEOrchestration::addDependencyEdge(TaskSuccessorMap &successors,
                                                Operation *producer,
                                                Operation *consumer) const {
  if (!producer || !consumer || producer == consumer) {
    return;
  }
  SmallVector<Operation *> &producer_successors = successors[producer];
  if (!llvm::is_contained(producer_successors, consumer)) {
    producer_successors.push_back(consumer);
  }
}

int SpatialDSEOrchestration::computeDependencyDepth(
    Operation *task, TaskSuccessorMap &successors,
    DenseMap<Operation *, int> &depth_cache,
    DenseSet<Operation *> &visiting) const {
  if (auto it = depth_cache.find(task); it != depth_cache.end()) {
    return it->second;
  }
  if (!visiting.insert(task).second) {
    return 0;
  }

  int max_child_depth = 0;
  for (Operation *successor : successors[task]) {
    max_child_depth = std::max(
        max_child_depth,
        computeDependencyDepth(successor, successors, depth_cache, visiting) +
            1);
  }

  visiting.erase(task);
  depth_cache[task] = max_child_depth;
  return max_child_depth;
}

TaskPriorityMap SpatialDSEOrchestration::computeRoutingCriticalPathPriority(
    func::FuncOp func) const {
  SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

  TaskSuccessorMap successors;
  for (TaskflowTaskOp task : tasks) {
    (void)successors[task.getOperation()];
  }

  for (TaskflowTaskOp consumer : tasks) {
    Operation *consumer_op = consumer.getOperation();
    auto addProducerFromValue = [&](Value value) {
      if (auto producer = value.getDefiningOp<TaskflowTaskOp>()) {
        addDependencyEdge(successors, producer.getOperation(), consumer_op);
      }
    };

    for (Value value_input : consumer.getValueInputs()) {
      addProducerFromValue(value_input);
    }
    for (Value operand : consumer->getOperands()) {
      addProducerFromValue(operand);
    }
  }

  TaskPriorityMap priority;
  DenseMap<Operation *, int> depth_cache;
  DenseSet<Operation *> visiting;
  for (TaskflowTaskOp task : tasks) {
    Operation *task_op = task.getOperation();
    priority[task_op] =
        computeDependencyDepth(task_op, successors, depth_cache, visiting);
  }
  return priority;
}

bool SpatialDSEOrchestration::runTaskOrchestration(func::FuncOp func) {
  if (!validateFixedShapeAttributes(func)) {
    return false;
  }

  TaskPriorityMap priority = computeRoutingCriticalPathPriority(func);
  TaskScheduler scheduler(grid_rows_, grid_cols_, mode_,
                          ShapeSelectionPolicy::FixedOrientation);
  if (!scheduler.schedule(func, priority)) {
    return false;
  }
  func.walk([](TaskflowTaskOp task) {
    task->removeAttr("amoeba.analytical_shape_orientation_fixed");
  });
  return true;
}

} // namespace taskflow
} // namespace mlir

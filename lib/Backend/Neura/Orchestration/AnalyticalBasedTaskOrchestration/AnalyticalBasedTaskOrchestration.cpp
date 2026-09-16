// Orchestrates a materialized analytical task candidate.

#include "Backend/Neura/Orchestration/AnalyticalBasedTaskOrchestration/AnalyticalBasedTaskOrchestration.h"

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

// Enumerates ordered rectangle dimensions whose area equals cgra_count and
// whose rows and columns fit the physical grid. Both 1x2 and 2x1 are returned
// when legal because they are distinct spatial rotations.
SmallVector<CgraShape> AnalyticalBasedTaskOrchestration::getRectangularShapes(
    int cgra_count, int grid_rows, int grid_cols) {
  SmallVector<CgraShape> shapes;
  if (cgra_count <= 0 || grid_rows <= 0 || grid_cols <= 0) {
    return shapes;
  }

  for (int rows = 1; rows <= grid_rows; ++rows) {
    if (cgra_count % rows != 0) {
      continue;
    }
    int cols = cgra_count / rows;
    if (cols <= grid_cols) {
      shapes.push_back({rows, cols, true, {}});
    }
  }
  return shapes;
}

// Infers a task's static execution count from its Taskflow counter forest.
// Each counter iterates over the half-open range [lower, upper) with a positive
// step. Nested counters multiply their counts; sibling chains and independent
// roots contribute their maximum chain count. The routine fails for dynamic
// bounds, empty ranges, malformed counter graphs, or int64_t overflow.
FailureOr<std::optional<int64_t>>
AnalyticalBasedTaskOrchestration::inferStaticTaskTripCount(TaskflowTaskOp task,
                                                           std::string &error) {
  // Collects every counter owned by the task. No counters means the manifest
  // may use the caller's default of one execution.
  SmallVector<TaskflowCounterOp> counters;
  task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
  if (counters.empty()) {
    return std::optional<int64_t>{};
  }

  if (!task.getBody().hasOneBlock()) {
    error = "task " + task.getTaskName().str() +
            " must contain exactly one block to infer a static trip count";
    return failure();
  }

  // Reconstructs the counter forest from parent-index SSA values. Roots have
  // no parent index; children are keyed by the counter index they reference.
  SmallVector<TaskflowCounterOp> roots;
  llvm::DenseMap<Value, SmallVector<TaskflowCounterOp>> children;
  for (TaskflowCounterOp counter : counters) {
    if (Value parent = counter.getParentIndex()) {
      children[parent].push_back(counter);
    } else {
      roots.push_back(counter);
    }
  }
  if (roots.empty()) {
    error = "task " + task.getTaskName().str() +
            " has counters but no root counter";
    return failure();
  }

  // Accepts only arith.constant index bounds so the inferred count is stable
  // before scheduling and independent of runtime values.
  auto constant_index = [](Value value) -> FailureOr<int64_t> {
    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
      return constant.value();
    }
    return failure();
  };

  // Computes ceil((upper - lower) / step) for one non-empty counter range.
  // The lower-bound check keeps upper - lower representable in int64_t.
  auto counter_trip_count =
      [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    FailureOr<int64_t> lower = constant_index(counter.getLowerBound());
    FailureOr<int64_t> upper = constant_index(counter.getUpperBound());
    FailureOr<int64_t> step = constant_index(counter.getStep());
    if (failed(lower) || failed(upper) || failed(step)) {
      return failure();
    }
    if (*step <= 0 || *upper <= *lower ||
        (*lower < 0 && *upper > std::numeric_limits<int64_t>::max() + *lower)) {
      return failure();
    }

    int64_t distance = *upper - *lower;
    return 1 + (distance - 1) / *step;
  };

  // Evaluates each nested chain with DFS. active detects a cycle in the
  // current path, while visited rejects a counter reached through multiple
  // parent paths. A parent's count multiplies the longest nested child chain.
  llvm::DenseSet<Operation *> active;
  llvm::DenseSet<Operation *> visited;
  std::function<FailureOr<int64_t>(TaskflowCounterOp)> chain_trip_count =
      [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    Operation *operation = counter.getOperation();
    if (!active.insert(operation).second || visited.contains(operation)) {
      error = "task " + task.getTaskName().str() +
              " has a cyclic or multiply referenced counter chain";
      return failure();
    }

    FailureOr<int64_t> count = counter_trip_count(counter);
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
        FailureOr<int64_t> child_count = chain_trip_count(child);
        if (failed(child_count)) {
          return failure();
        }
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

  // Multiple roots represent independent counter chains, so the task count is
  // the maximum root-chain count rather than their product.
  int64_t total = 1;
  for (TaskflowCounterOp root : roots) {
    FailureOr<int64_t> root_count = chain_trip_count(root);
    if (failed(root_count)) {
      return failure();
    }
    total = std::max(total, *root_count);
  }

  // Every collected counter must be reachable from exactly one root.
  if (visited.size() != counters.size()) {
    error = "task " + task.getTaskName().str() +
            " has a counter disconnected from every root";
    return failure();
  }
  return std::optional<int64_t>{total};
}

namespace {

// Parses the materialized rowsxcols spelling without applying another rotation.
// The ordered dimensions must be positive, fit the physical grid, and have an
// area equal to cgra_count.
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

// Verifies that the function and every task carry a complete materialized
// spatial candidate. The fixed-rotation marker distinguishes a deliberate
// rows-by-columns choice from a shape that the default scheduler may rotate.
// Diagnostics are emitted at the function or offending task, and validation
// fails closed before any placement metadata is published.
bool AnalyticalBasedTaskOrchestration::validateFixedShapeAttributes(
    func::FuncOp func) const {
  auto candidate_id =
      func->getAttrOfType<StringAttr>("analytical_task_candidate_id");
  if (!candidate_id || candidate_id.getValue().empty()) {
    func.emitError()
        << "analytical-based task orchestration requires a non-empty "
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
      task.emitError() << "analytical-based task orchestration requires "
                          "amoeba.analytical_shape_orientation_fixed";
      valid = false;
      return;
    }

    auto cgra_count = task->getAttrOfType<IntegerAttr>("cgra_count");
    if (!cgra_count || cgra_count.getInt() <= 0 ||
        cgra_count.getInt() > std::numeric_limits<int>::max()) {
      task.emitError() << "analytical-based task orchestration requires a "
                          "positive cgra_count";
      valid = false;
      return;
    }

    auto cgra_shape = task->getAttrOfType<StringAttr>("cgra_shape");
    if (!cgra_shape || cgra_shape.getValue().empty()) {
      task.emitError() << "analytical-based task orchestration requires a "
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

// Records one producer-to-consumer edge in the task successor graph. Invalid
// endpoints, self-edges, and duplicate edges cannot affect the priority map.
void AnalyticalBasedTaskOrchestration::addDependencyEdge(
    TaskSuccessorMap &successors, Operation *producer,
    Operation *consumer) const {
  if (!producer || !consumer || producer == consumer) {
    return;
  }
  SmallVector<Operation *> &producer_successors = successors[producer];
  if (!llvm::is_contained(producer_successors, consumer)) {
    producer_successors.push_back(consumer);
  }
}

// Returns the longest number of dependency edges from task to a graph sink.
// Memoizes completed depths so each acyclic subgraph is traversed once. A back
// edge contributes zero here; TaskScheduler later diagnoses dependency cycles.
int AnalyticalBasedTaskOrchestration::computeDependencyDepth(
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

// Builds task dependencies from both Taskflow value inputs and the complete
// operand list, then uses longest-to-sink depth as the scheduler priority.
// Larger values place producers on the routing critical path first.
TaskPriorityMap
AnalyticalBasedTaskOrchestration::computeRoutingCriticalPathPriority(
    func::FuncOp func) const {
  SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });

  TaskSuccessorMap successors;
  for (TaskflowTaskOp task : tasks) {
    (void)successors[task.getOperation()];
  }

  for (TaskflowTaskOp consumer : tasks) {
    Operation *consumer_op = consumer.getOperation();
    // Adds an edge from the Taskflow task defining the value to this consumer.
    auto add_producer_from_value = [&](Value value) {
      if (auto producer = value.getDefiningOp<TaskflowTaskOp>()) {
        addDependencyEdge(successors, producer.getOperation(), consumer_op);
      }
    };

    for (Value value_input : consumer.getValueInputs()) {
      add_producer_from_value(value_input);
    }
    for (Value operand : consumer->getOperands()) {
      add_producer_from_value(operand);
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

// Validates and schedules one materialized candidate. FixedOrientation makes
// the scheduler preserve each supplied rotation instead of generating other
// rotations or fallback shapes. The temporary marker is removed only after
// scheduling succeeds.
bool AnalyticalBasedTaskOrchestration::runTaskOrchestration(func::FuncOp func) {
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

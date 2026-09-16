// Orchestrates a materialized analytical task candidate.

#ifndef AMOEBA_ANALYTICAL_BASED_TASK_ORCHESTRATION_H
#define AMOEBA_ANALYTICAL_BASED_TASK_ORCHESTRATION_H

#include "Backend/Neura/Orchestration/Orchestration.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace taskflow {

// Orchestrates a materialized analytical task candidate.
//
// This strategy consumes one candidate after candidate-space exploration; it
// does not construct or search that space. The current candidate manifest
// records one spatial rectangle with a fixed rotation per task.
// SchedulingMode independently controls whether those tasks may time-share
// CGRAs while the selected spatial candidate executes.
//
// TODO: Consume temporal candidate decisions after a temporal candidate space
// and its manifest representation are introduced.
class AnalyticalBasedTaskOrchestration : public Orchestration {
public:
  AnalyticalBasedTaskOrchestration(
      int grid_rows = kCgraGridRows, int grid_cols = kCgraGridCols,
      SchedulingMode mode = SchedulingMode::SpatialTemporal)
      : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode) {}

  bool runTaskOrchestration(mlir::func::FuncOp func) override;

  // Enumerates legal rectangular rotations for one physical-CGRA count. Each
  // ordered rows-by-columns pair represents a distinct spatial choice. The
  // analytical scheduler later uses FixedOrientation, so it does not rotate
  // the selected choice again.
  static llvm::SmallVector<CgraShape>
  getRectangularShapes(int cgra_count, int grid_rows = kCgraGridRows,
                       int grid_cols = kCgraGridCols);

  // Derives the static execution count from constant Taskflow counter chains.
  // Returns std::nullopt when the task has no counters; callers interpret that
  // case as one execution. Returns failure for malformed or dynamic chains.
  static FailureOr<std::optional<int64_t>>
  inferStaticTaskTripCount(TaskflowTaskOp task, std::string &error);

  std::string getName() const override {
    return "analytical-based-task-orchestration";
  }

private:
  using TaskSuccessorMap =
      llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>>;

  // Validates the complete materialized spatial assignment before scheduling.
  bool validateFixedShapeAttributes(func::FuncOp func) const;

  // Adds one dependency edge while ignoring invalid, self, and duplicate edges.
  void addDependencyEdge(TaskSuccessorMap &successors, Operation *producer,
                         Operation *consumer) const;

  // Computes the longest successor path from one task with memoized DFS.
  int computeDependencyDepth(Operation *task, TaskSuccessorMap &successors,
                             llvm::DenseMap<Operation *, int> &depth_cache,
                             llvm::DenseSet<Operation *> &visiting) const;

  // Assigns higher priority to tasks farther from a dependency-graph sink.
  TaskPriorityMap computeRoutingCriticalPathPriority(func::FuncOp func) const;

  int grid_rows_;
  int grid_cols_;
  SchedulingMode mode_;
};

} // namespace taskflow
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_BASED_TASK_ORCHESTRATION_H

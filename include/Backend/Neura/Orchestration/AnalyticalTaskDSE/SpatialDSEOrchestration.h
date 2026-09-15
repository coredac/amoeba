// Orchestration of a materialized spatial analytical DSE candidate.

#ifndef AMOEBA_SPATIAL_DSE_ORCHESTRATION_H
#define AMOEBA_SPATIAL_DSE_ORCHESTRATION_H

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

// Concrete orchestration strategy for a materialized spatial DSE candidate.
//
// Its design space assigns each task a fixed-orientation rectangular region of
// the physical CGRA grid. The materializer records one complete assignment on
// the function and its tasks; this strategy validates and places that exact
// assignment.
class SpatialDSEOrchestration : public Orchestration {
public:
  SpatialDSEOrchestration(int grid_rows = kCgraGridRows,
                          int grid_cols = kCgraGridCols,
                          SchedulingMode mode = SchedulingMode::SpatialTemporal)
      : grid_rows_(grid_rows), grid_cols_(grid_cols), mode_(mode) {}

  bool runTaskOrchestration(mlir::func::FuncOp func) override;

  // Enumerates the fixed-orientation rectangular shape choices explored by
  // this strategy for one physical-CGRA count.
  static llvm::SmallVector<CgraShape>
  getRectangularShapes(int cgraCount, int gridRows = kCgraGridRows,
                       int gridCols = kCgraGridCols);

  // Derives the execution count used by this strategy from constant Taskflow
  // counter chains. A task without a counter represents one execution.
  static FailureOr<std::optional<int64_t>>
  inferStaticTaskTripCount(TaskflowTaskOp task, std::string &error);

  std::string getName() const override { return "analytical-dse-spatial"; }

private:
  using TaskSuccessorMap =
      llvm::DenseMap<Operation *, llvm::SmallVector<Operation *>>;

  bool validateFixedShapeAttributes(func::FuncOp func) const;

  void addDependencyEdge(TaskSuccessorMap &successors, Operation *producer,
                         Operation *consumer) const;

  int computeDependencyDepth(Operation *task, TaskSuccessorMap &successors,
                             llvm::DenseMap<Operation *, int> &depth_cache,
                             llvm::DenseSet<Operation *> &visiting) const;

  TaskPriorityMap computeRoutingCriticalPathPriority(func::FuncOp func) const;

  int grid_rows_;
  int grid_cols_;
  SchedulingMode mode_;
};

} // namespace taskflow
} // namespace mlir

#endif // AMOEBA_SPATIAL_DSE_ORCHESTRATION_H

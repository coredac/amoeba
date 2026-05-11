//===- OrchestrateTaskOnCgraPass.cpp - Task to CGRA Orchestration Pass ---===//
//
// Implements the orchestrate-task-on-cgra pass, which maps Taskflow tasks
// onto a 2D multi-CGRA grid array:
// 1. Places tasks with SSA dependencies (producer-consumer pairs) on
//    adjacent CGRAs to enable direct data forwarding.
// 2. Adds spatial-temporal scheduling (start_time, duration) per task.
// 3. Assigns memrefs to SRAMs (each MemRef is assigned to exactly one SRAM,
//    determined by proximity to the task that first accesses it).
//
// Implementation: RoutingCriticalPathAllocation in
// lib/TaskflowDialect/Allocation/RoutingCriticalPathAllocation.cpp.
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/Allocation/RoutingCriticalPathAllocation.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

using namespace mlir;
using namespace mlir::taskflow;

namespace {

struct OrchestrateTaskOnCgraPass
    : public OrchestrateTaskOnCgraBase<OrchestrateTaskOnCgraPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(OrchestrateTaskOnCgraPass)

  void runOnOperation() override {
    AllocationMode mode = (allocationMode == "spatial")
                              ? AllocationMode::Spatial
                              : AllocationMode::SpatialTemporal;
    RoutingCriticalPathAllocation strategy(kCgraGridRows, kCgraGridCols, mode);
    strategy.runAllocation(getOperation());
  }
};

} // namespace

namespace mlir {
namespace taskflow {

std::unique_ptr<Pass> createOrchestrateTaskOnCgraPass() {
  return std::make_unique<OrchestrateTaskOnCgraPass>();
}

} // namespace taskflow
} // namespace mlir

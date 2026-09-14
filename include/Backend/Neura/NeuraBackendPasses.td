// NeuraBackendPasses.td - Passes owned by the Neura backend

#ifndef AMOEBA_BACKEND_NEURA_PASSES_TD
#define AMOEBA_BACKEND_NEURA_PASSES_TD

include "mlir/Pass/PassBase.td"

def ConvertTaskflowToNeura : Pass<"convert-taskflow-to-neura", "ModuleOp"> {
  let summary = "Convert Taskflow operations to neura.kernel";
  let description = [{
    Converts taskflow.hyperblock operations and their counters into
    neura.kernel operations suitable for CGRA tile-array mapping.
  }];
  let constructor =
      "::mlir::amoeba::neura::createConvertTaskflowToNeuraPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::affine::AffineDialect"];
}

def ConstructHyperblockFromTask
    : Pass<"construct-hyperblock-from-task", "func::FuncOp"> {
  let summary = "Construct Taskflow hyperblocks for the Neura backend";
  let description = [{
    Selects one of two transforms based on the Neura architecture.
    With counter support, the pass detects perfect nested loop bands.
    Without counter support, it wraps the innermost loop as a hyperblock.
  }];
  let constructor =
      "::mlir::amoeba::neura::createConstructHyperblockFromTaskPass()";
}

def ClassifyTaskAndCounter : Pass<"classify-task-and-counter", "ModuleOp"> {
  let summary = "Classify tasks and counters for Neura runtime management";
  let description = [{
    Annotates taskflow.task and taskflow.counter operations with the
    structural and runtime-management information consumed by Neura.
  }];
  let constructor =
      "::mlir::amoeba::neura::createClassifyTaskAndCounterPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def OrchestrateTasksOnAccelerators
    : Pass<"orchestrate-tasks-on-accelerators", "func::FuncOp"> {
  let summary = "Orchestrate Taskflow tasks on a Neura multi-CGRA grid";
  let description = [{
    Computes task resource allocation, placement, and scheduling for the
    Neura multi-CGRA architecture. Selects an orchestration strategy and
    invokes the shared scheduler. The routing-critical-path strategy is the
    default and supports spatial and spatial-temporal scheduling.
  }];
  let options = [Option<
      "schedulingMode", "scheduling-mode", "std::string",
      /*default=*/"\"spatial-temporal\"",
      "Task scheduling mode: 'spatial' (one task per CGRA) or "
      "'spatial-temporal' (time-multiplexes CGRAs).">,
    Option<
      "orchestrationStrategy", "orchestration-strategy", "std::string",
      /*default=*/"\"routing-critical-path\"",
      "Task orchestration strategy: 'routing-critical-path' (default).">];
  let constructor =
      "::mlir::amoeba::neura::createOrchestrateTasksOnAcceleratorsPass()";
}

def FuseTask : Pass<"fuse-task", "func::FuncOp"> {
  let summary = "Fuse Taskflow tasks using Neura profitability metrics";
  let description = [{
    Applies producer-consumer and sibling fusion strategies. Profitability is
    evaluated with Neura lowering and MII metrics.
  }];
  let constructor = "::mlir::amoeba::neura::createFuseTaskPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect",
                           "mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ResourceAwareTaskOptimization
    : Pass<"resource-aware-task-optimization", "func::FuncOp"> {
  let summary =
      "Optimize Taskflow resource allocation for the Neura CGRA backend";
  let description = [{
    Balances pipeline latency and fuses independent tasks for Neura CGRA
    utilization. Compiled estimation runs the Neura lowering and mapping
    flow; analytical estimation uses ResMII/RecMII estimates.
  }];
  let options = [Option<
      "estimationMode", "estimation-mode", "std::string",
      /*default=*/"\"compiled\"",
      "Profiling mode: 'compiled' runs Neura lowering and mapping; "
      "'analytical' uses ResMII/RecMII estimates.">];
  let constructor =
      "::mlir::amoeba::neura::createResourceAwareTaskOptimizationPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect"];
}

#endif // AMOEBA_BACKEND_NEURA_PASSES_TD

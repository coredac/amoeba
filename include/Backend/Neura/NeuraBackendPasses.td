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
  let constructor = "::mlir::amoeba::neura::createConvertTaskflowToNeuraPass()";
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
  let constructor = "::mlir::amoeba::neura::createClassifyTaskAndCounterPass()";
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

def AnalyzeAnalyticalLowerBound
    : Pass<"analyze-analytical-lower-bound", "ModuleOp"> {
  let summary = "Compute a mapper-compatible lower bound without mapping";
  let description = [{
    Computes the analytical initiation-interval lower bound used by Neura's
    heuristic mapper for exactly one standalone task DFG. x-tiles and y-tiles
    select the concrete mapper rectangle. The combined lower bound and its
    RecMII and ResMII components are written as module attributes for the
    external ML feature adapter. The pass never places or routes an operation.
  }];
  let options = [Option<"xTiles", "x-tiles", "int64_t", /*default=*/"0",
                         "Concrete mapper tile columns; must be positive.">,
                 Option<"yTiles", "y-tiles", "int64_t", /*default=*/"0",
                         "Concrete mapper tile rows; must be positive.">];
  let constructor =
      "::mlir::amoeba::neura::createAnalyzeAnalyticalLowerBoundPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::neura::NeuraDialect"];
}

def EnumerateAnalyticalTaskCandidates
    : Pass<"enumerate-analytical-task-candidates", "ModuleOp"> {
  let summary = "Freeze all concurrently packable static task shapes";
  let description = [{
    Enumerates every ordered assignment of fixed-orientation rectangular
    per-task CGRA shapes that admits a simultaneous, non-overlapping placement
    on the physical multi-CGRA grid, then writes a JSONL manifest. Exact
    placement feasibility is checked with backtracking; concrete placement
    coordinates remain a downstream heuristic choice. Constant counter chains
    record their exact trip count, while symbol-bound chains are preserved as
    symbol-dynamic. Counter bound information is read from the
    `counter_dynamism` attributes produced by `classify-task-and-counter`,
    which must run before this pass. The pass does not score candidates or
    invoke the mapper. Fusion, fission, tiling, placement, and temporal order
    are fixed.
  }];
  let options =
      [Option<
           "functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
       Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
              "Path to the candidate JSONL manifest.">];
  let constructor =
      "::mlir::amoeba::neura::createEnumerateAnalyticalTaskCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def MaterializeAnalyticalTaskCandidate
    : Pass<"materialize-analytical-task-candidate", "ModuleOp"> {
  let summary = "Materialize one static task-shape candidate";
  let description = [{
    Validates a complete candidate manifest, selects one candidate by its
    sequential ID, and writes its cgra_count and cgra_shape decisions to
    Taskflow tasks. The selected rectangle orientation is locked so resource
    allocation cannot silently replace it. This pass does not invoke the
    mapper. Taskflow counters must first be annotated by
    `classify-task-and-counter`, matching the enumeration pipeline that
    produced the manifest.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
    Option<"candidateFile", "candidates", "std::string", /*default=*/"\"\"",
           "Path to the candidate JSONL manifest.">,
    Option<"candidateId", "candidate-id", "std::string", /*default=*/"\"\"",
           "Sequential candidate ID to materialize.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createMaterializeAnalyticalTaskCandidatePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def ScoreAnalyticalTaskCandidates
    : Pass<"score-analytical-task-candidates", "ModuleOp"> {
  let summary = "Score every candidate in a frozen analytical-task manifest";
  let description = [{
    Validates and scores every candidate from a complete manifest using a
    provenance-bound external ML II/startup cost catalogue. Repeated queries
    share a stable task-body, architecture, and oriented-shape cache. Only
    after complete manifest traversal is a deterministic top-k shortlist
    emitted. The current score is the maximum per-task compute duration;
    communication and temporal scheduling remain outside this pass. A
    symbol-dynamic trip count must be concretely bound before ranking.
  }];
  let options = [
    Option<"functionName", "function", "std::string", /*default=*/"\"\"",
           "Taskflow function to process; inferred when exactly one exists.">,
    Option<"candidateFile", "candidates", "std::string", /*default=*/"\"\"",
           "Path to the frozen candidate JSONL manifest.">,
    Option<"costFile", "cost-file", "std::string", /*default=*/"\"\"",
           "Path to the task-shape ML cost catalogue.">,
    Option<"outputFile", "output", "std::string", /*default=*/"\"\"",
           "Path to the score JSONL manifest.">,
    Option<"topK", "top-k", "int64_t", /*default=*/"1",
           "Candidates to shortlist; zero selects every valid candidate.">
  ];
  let constructor =
      "::mlir::amoeba::neura::createScoreAnalyticalTaskCandidatesPass()";
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect",
                           "mlir::taskflow::TaskflowDialect"];
}

def FuseTask : Pass<"fuse-task", "func::FuncOp"> {
  let summary = "Fuse Taskflow tasks using Neura profitability metrics";
  let description = [{
    Applies producer-consumer and sibling fusion strategies. Profitability is
    evaluated with Neura lowering and MII metrics.
  }];
  let constructor = "::mlir::amoeba::neura::createFuseTaskPass()";
  let dependentDialects = ["mlir::LLVM::LLVMDialect", "mlir::func::FuncDialect",
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

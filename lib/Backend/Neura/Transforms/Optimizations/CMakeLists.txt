get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaNeuraOptimizations
  AnalyticalTaskCandidateManifest.cpp
  AnalyticalTaskCandidateSpace.cpp
  AnalyticalTaskCostCatalog.cpp
  AnalyzeAnalyticalLowerBoundPass.cpp
  EnumerateAnalyticalTaskCandidatesPass.cpp
  FuseTaskPass.cpp
  MaterializeAnalyticalTaskCandidatePass.cpp
  ResourceAwareTaskOptimizationPass.cpp
  ScoreAnalyticalTaskCandidatesPass.cpp

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraBackendOptions
  MLIRAmoebaNeuraConversion
  MLIRAmoebaNeuraOrchestrationUtil
  MLIRAmoebaNeuraTransforms
  MLIRTaskflow
  MLIRNeura
  MLIRNeuraTransforms
  MLIRNeuraConversion
  MLIRIR
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTransforms
  ${dialect_libs}
  LLVMSupport
)

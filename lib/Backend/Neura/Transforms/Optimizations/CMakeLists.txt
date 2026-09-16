get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaNeuraOptimizations
  AnalyticalTaskCandidateCommon.cpp
  EnumerateAnalyticalTaskCandidatesPass.cpp
  FuseTaskPass.cpp
  ResourceAwareTaskOptimizationPass.cpp
  SpatialTaskCandidateSpace.cpp

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraAnalyticalBasedTaskOrchestration
  MLIRAmoebaNeuraBackendOptions
  MLIRAmoebaNeuraConversion
  MLIRAmoebaNeuraTransforms
  MLIRArithDialect
  MLIRFuncDialect
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

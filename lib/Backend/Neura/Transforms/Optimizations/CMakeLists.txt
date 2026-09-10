get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaNeuraOptimizations
  AnalyticalTaskCandidateManifest.cpp
  AnalyticalTaskCandidateSpace.cpp
  EnumerateAnalyticalTaskCandidatesPass.cpp
  FuseTaskPass.cpp
  MaterializeAnalyticalTaskCandidatePass.cpp
  ResourceAwareTaskOptimizationPass.cpp

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraBackendOptions
  MLIRAmoebaNeuraConversion
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

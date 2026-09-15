get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaNeuraAnalyticalTaskDSE
  AnalyticalTaskCandidateCommon.cpp
  EnumerateAnalyticalTaskCandidatesPass.cpp
  SpatialDSEOrchestration.cpp
  SpatialTaskCandidateSpace.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraBackendOptions
  MLIRAmoebaNeuraOrchestrationUtil
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRNeura
  MLIRParser
  MLIRPass
  MLIRSupport
  MLIRTaskflow
  ${dialect_libs}
  LLVMSupport
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaNeuraTransforms
  ConstructHyperblockFromTaskPass.cpp
  ClassifyTaskAndCounterPass.cpp
  OrchestrateTasksOnAcceleratorsPass.cpp

  DEPENDS
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraOrchestration
  MLIRTaskflow
  MLIRNeura
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  ${dialect_libs}
  LLVMSupport
)

add_subdirectory(Optimizations)

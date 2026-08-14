get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRAmoebaNeuraOrchestration
  orchestration_utils.cpp
  RoutingCriticalPathOrchestration/RoutingCriticalPathOrchestration.cpp

  LINK_LIBS PUBLIC
  MLIRTaskflow
  MLIRIR
  MLIRPass
  MLIRSupport
  ${dialect_libs}
  LLVMSupport
)

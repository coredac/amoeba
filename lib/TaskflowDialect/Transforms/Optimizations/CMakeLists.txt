get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)

add_mlir_library(MLIRTaskflowTransforms
  AffineLoopTreeSerializationPass.cpp
  AffineLoopPerfectionPass.cpp
  MemoryAccessStreamingFusion.cpp

  DEPENDS
  MLIRTaskflowTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRTaskflow
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransforms
  ${dialect_libs}
  LLVMSupport
)

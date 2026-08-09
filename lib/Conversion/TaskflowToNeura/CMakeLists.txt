include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_mlir_conversion_library(MLIRTaskflowToNeuraPass
  TaskflowToNeuraPass.cpp

  DEPENDS
  MLIRAmoebaConversionIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLLVMDialect
  MLIRTaskflow
  MLIRNeura
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
)

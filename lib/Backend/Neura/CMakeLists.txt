add_subdirectory(Conversion)
add_subdirectory(Orchestration)
add_subdirectory(Transforms)

add_mlir_library(MLIRAmoebaNeuraBackend
  NeuraBackend.cpp
  NeuraBackendPasses.cpp

  DEPENDS
  MLIRAmoebaConversionIncGen
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaConversion
  MLIRAmoebaNeuraConversion
  MLIRAmoebaNeuraTransforms
  MLIRAmoebaNeuraOptimizations
  MLIRTaskflow
  MLIRNeura
  MLIRNeuraTransforms
  MLIRNeuraConversion
  MLIRNUERAOptimization
  MLIRIR
  MLIRPass
  MLIRSupport
  LLVMSupport
)

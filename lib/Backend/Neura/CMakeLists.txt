add_mlir_library(MLIRAmoebaNeuraBackendOptions
  NeuraBackendOptions.cpp

  PARTIAL_SOURCES_INTENDED

  LINK_LIBS PUBLIC
  LLVMSupport
)

add_subdirectory(Conversion)
add_subdirectory(Orchestration)
add_subdirectory(Transforms)

add_mlir_library(MLIRAmoebaNeuraBackend
  NeuraBackend.cpp
  NeuraBackendPasses.cpp

  PARTIAL_SOURCES_INTENDED

  DEPENDS
  MLIRAmoebaConversionIncGen
  MLIRAmoebaNeuraBackendPassesIncGen

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraBackendOptions
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

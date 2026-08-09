add_mlir_library(MLIRAmoebaBackends
  Backends.cpp
  Neura/NeuraBackend.cpp

  LINK_LIBS PUBLIC
  MLIRNeura
  MLIRNeuraTransforms
  MLIRNeuraHwAgnosticOpt
  MLIRNeuraHwSpecificOpt
  MLIRNeuraConversion
  MLIRIR
  MLIRPass
  MLIRSupport
  LLVMSupport
)

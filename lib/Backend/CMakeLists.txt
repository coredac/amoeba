add_subdirectory(Neura)

add_mlir_library(MLIRAmoebaBackends
  Backends.cpp

  LINK_LIBS PUBLIC
  MLIRAmoebaNeuraBackend
  MLIRIR
  MLIRSupport
)

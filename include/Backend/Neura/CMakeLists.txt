set(LLVM_TARGET_DEFINITIONS NeuraBackendPasses.td)
mlir_tablegen(NeuraBackendPasses.h.inc --gen-pass-decls -name NeuraBackend)
add_public_tablegen_target(MLIRAmoebaNeuraBackendPassesIncGen)

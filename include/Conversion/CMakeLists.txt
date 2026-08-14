set(LLVM_TARGET_DEFINITIONS AmoebaConversionPasses.td)
mlir_tablegen(AmoebaConversionPasses.h.inc --gen-pass-decls -name AmoebaConversion)
add_public_tablegen_target(MLIRAmoebaConversionIncGen)

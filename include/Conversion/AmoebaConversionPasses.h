// AmoebaConversionPasses.h - Header file for Amoeba conversion passes

#ifndef AMOEBA_CONVERSION_PASSES_H
#define AMOEBA_CONVERSION_PASSES_H
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include <memory>

namespace mlir {

// Passes defined in AmoebaConversionPasses.td.
#define GEN_PASS_DECL
#include "Conversion/AmoebaConversionPasses.h.inc"

// Taskflow conversion passes.
std::unique_ptr<mlir::Pass> createConvertAffineToTaskflowPass();
std::unique_ptr<mlir::Pass> createConvertTaskflowToNeuraPass();

// MemRef subview and copy conversion passes.
std::unique_ptr<mlir::Pass> createFoldSubViewPass();
std::unique_ptr<mlir::Pass> createConvertCopyToAffineLoopsPass();

#define GEN_PASS_REGISTRATION
#include "Conversion/AmoebaConversionPasses.h.inc"

} // namespace mlir

#endif // AMOEBA_CONVERSION_PASSES_H

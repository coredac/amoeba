//===- NeuraBackend.cpp - Neura backend integration ----------------------===//

#include "Backend/Neura/NeuraBackend.h"
#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/NeuraBackendPasses.h"

#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Util/ArchParser.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/ErrorHandling.h"

using mlir::neura::Architecture;
using mlir::neura::util::ArchParser;

const Architecture &mlir::neura::getArchitecture() {
  static Architecture architecture = []() {
    ArchParser parser(mlir::amoeba::getNeuraArchitectureSpecFile());
    auto result = parser.getArchitecture();
    if (failed(result))
      llvm::report_fatal_error("[neura-backend] Failed to get architecture.");
    return std::move(*result);
  }();
  return architecture;
}

const std::string &mlir::neura::getLatencySpecFile() {
  return mlir::amoeba::getNeuraLatencySpecFile();
}

void mlir::amoeba::registerNeuraBackend(DialectRegistry &registry) {
  registry.insert<mlir::neura::NeuraDialect>();
  mlir::neura::registerPasses();
  mlir::registerNeuraConversionPasses();
  mlir::amoeba::neura::registerNeuraBackendPasses();
  mlir::amoeba::neura::registerTaskflowConversionPassPipeline();
}

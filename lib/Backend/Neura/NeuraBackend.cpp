//===- NeuraBackend.cpp - Neura backend integration ----------------------===//

#include "Backend/Neura/NeuraBackend.h"

#include "Conversion/NeuraConversionPasses.h"
#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/NeuraDialect.h"
#include "NeuraDialect/NeuraPasses.h"
#include "NeuraDialect/Util/ArchParser.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorHandling.h"

using mlir::neura::Architecture;
using mlir::neura::util::ArchParser;

namespace {

llvm::cl::opt<std::string> neuraArchitectureSpec(
    "neura-architecture-spec",
    llvm::cl::desc("Path to the Neura architecture specification"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::alias
    architectureSpecAlias("architecture-spec",
                          llvm::cl::desc("Alias for --neura-architecture-spec"),
                          llvm::cl::aliasopt(neuraArchitectureSpec));

llvm::cl::opt<std::string>
    neuraLatencySpec("neura-latency-spec",
                     llvm::cl::desc("Path to the Neura latency specification"),
                     llvm::cl::value_desc("path"), llvm::cl::init(""));

llvm::cl::alias
    latencySpecAlias("latency-spec",
                     llvm::cl::desc("Alias for --neura-latency-spec"),
                     llvm::cl::aliasopt(neuraLatencySpec));

} // namespace

const Architecture &mlir::neura::getArchitecture() {
  static Architecture architecture = []() {
    ArchParser parser(neuraArchitectureSpec.getValue());
    auto result = parser.getArchitecture();
    if (failed(result))
      llvm::report_fatal_error("[neura-backend] Failed to get architecture.");
    return std::move(*result);
  }();
  return architecture;
}

const std::string &mlir::neura::getLatencySpecFile() {
  return neuraLatencySpec.getValue();
}

void mlir::amoeba::registerNeuraBackend(DialectRegistry &registry) {
  registry.insert<mlir::neura::NeuraDialect>();
  mlir::neura::registerPasses();
  mlir::registerNeuraConversionPasses();
}

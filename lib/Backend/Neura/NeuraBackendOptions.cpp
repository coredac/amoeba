//===- NeuraBackendOptions.cpp - shared Neura CLI options ----------------===//

#include "Backend/Neura/NeuraBackendOptions.h"

#include "llvm/Support/CommandLine.h"

// Keep the option storage separate from NeuraBackend.cpp. Optimization passes
// need these values, while the backend itself links the optimization library;
// defining the options in the backend would introduce a circular library
// dependency.
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

const std::string &mlir::amoeba::getNeuraArchitectureSpecFile() {
  return neuraArchitectureSpec.getValue();
}

const std::string &mlir::amoeba::getNeuraLatencySpecFile() {
  return neuraLatencySpec.getValue();
}

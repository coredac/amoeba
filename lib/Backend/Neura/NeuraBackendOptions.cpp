//===- NeuraBackendOptions.cpp - shared Neura CLI options ----------------===//

#include "Backend/Neura/NeuraBackendOptions.h"

#include "llvm/Support/CommandLine.h"

// Keep command-line option storage in a small dependency-neutral library.
// Candidate-enumeration code in MLIRAmoebaNeuraOptimizations needs the
// architecture path, while MLIRAmoebaNeuraBackend already links that
// optimization library. If NeuraBackend.cpp owned the option objects, the
// optimization library would have to link back to the backend to read them,
// creating Backend -> Optimizations -> Backend. Both libraries instead depend
// downward on MLIRAmoebaNeuraBackendOptions through these accessors.
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

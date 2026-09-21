// NeuraBackendPasses.h - Passes owned by the Neura backend

#ifndef AMOEBA_BACKEND_NEURA_PASSES_H
#define AMOEBA_BACKEND_NEURA_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include <memory>

namespace mlir {
namespace amoeba {
namespace neura {

// Registers the existing affine -> Taskflow -> Neura convenience pipeline.
void registerTaskflowConversionPassPipeline();

// Passes defined in NeuraBackendPasses.td.
#define GEN_PASS_DECL
#include "Backend/Neura/NeuraBackendPasses.h.inc"

std::unique_ptr<Pass> createConvertTaskflowToNeuraPass();
std::unique_ptr<Pass> createConstructHyperblockFromTaskPass();
std::unique_ptr<Pass> createClassifyTaskAndCounterPass();
std::unique_ptr<Pass> createOrchestrateTasksOnAcceleratorsPass();
std::unique_ptr<Pass> createEnumerateAnalyticalTaskCandidatesPass();
std::unique_ptr<Pass> createFuseTaskPass();
std::unique_ptr<Pass> createResourceAwareTaskOptimizationPass();

#define GEN_PASS_REGISTRATION
#include "Backend/Neura/NeuraBackendPasses.h.inc"

} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_PASSES_H

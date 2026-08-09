#include "Backend/Neura/NeuraBackendPasses.h"

#include "Conversion/AmoebaConversionPasses.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

using namespace mlir;

void mlir::amoeba::neura::registerTaskflowConversionPassPipeline() {
  PassPipelineRegistration<>(
      "taskflow-conversion",
      "Convert affine IR through Taskflow to the Neura backend.",
      [](OpPassManager &pm) {
        pm.addPass(mlir::createConvertAffineToTaskflowPass());
        pm.addPass(
            mlir::amoeba::neura::createConstructHyperblockFromTaskPass());
        pm.addPass(
            mlir::amoeba::neura::createClassifyTaskAndCounterPass());
        pm.addPass(
            mlir::amoeba::neura::createConvertTaskflowToNeuraPass());
      });
}

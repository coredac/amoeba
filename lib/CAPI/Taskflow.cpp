#include "Taskflow-C/Taskflow.h"

#include "Backend/Backends.h"
#include "Backend/Neura/NeuraBackend.h"
#include "Conversion/AmoebaConversionPasses.h"
#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowPasses.h"
#include "mlir/CAPI/Registration.h"
#include "mlir/IR/DialectRegistry.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Taskflow, taskflow,
                                      mlir::taskflow::TaskflowDialect)

void mlirTaskflowSetNeuraArchitectureSpec(
    MlirStringRef architectureSpec) {
  mlir::amoeba::setNeuraArchitectureSpec(
      std::string(architectureSpec.data, architectureSpec.length));
}

void mlirTaskflowRegisterPasses() {
  mlir::registerAmoebaConversionPasses();
  mlir::taskflow::registerPasses();
  mlir::taskflow::registerTosaToAffineConversionPassPipeline();
  mlir::taskflow::registerLinalgToAffineConversionPassPipeline();

  mlir::DialectRegistry registry;
  mlir::amoeba::registerBackends(registry);
}
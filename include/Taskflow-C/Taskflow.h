#ifndef TASKFLOW_C_TASKFLOW_H
#define TASKFLOW_C_TASKFLOW_H

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Taskflow, taskflow);

// Registers Taskflow and backend passes.
MLIR_CAPI_EXPORTED void mlirTaskflowRegisterPasses(void);

// Sets the architecture specification used by Neura backend passes.
MLIR_CAPI_EXPORTED void
mlirTaskflowSetNeuraArchitectureSpec(MlirStringRef architectureSpec);

#ifdef __cplusplus
}
#endif

#endif // TASKFLOW_C_TASKFLOW_H

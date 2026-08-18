#ifndef TASKFLOW_C_TASKFLOW_H
#define TASKFLOW_C_TASKFLOW_H

#include "mlir-c/IR.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Taskflow, taskflow);

#ifdef __cplusplus
}
#endif

#endif // TASKFLOW_C_TASKFLOW_H

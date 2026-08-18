#include "Taskflow-C/Taskflow.h"

#include "TaskflowDialect/TaskflowDialect.h"
#include "mlir/CAPI/Registration.h"

MLIR_DEFINE_CAPI_DIALECT_REGISTRATION(Taskflow, taskflow,
                                      mlir::taskflow::TaskflowDialect)

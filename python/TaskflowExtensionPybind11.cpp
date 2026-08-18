#include "Taskflow-C/Taskflow.h"

#include "mlir/Bindings/Python/PybindAdaptors.h"

using namespace mlir::python::adaptors;

PYBIND11_MODULE(_TaskflowExtensionPybind11, module) {
  auto taskflow_module = module.def_submodule("taskflow");

  taskflow_module.def(
      "register_dialect",
      [](MlirContext context, bool load) {
        MlirDialectHandle handle = mlirGetDialectHandle__taskflow__();

        mlirDialectHandleRegisterDialect(handle, context);

        if (load) {
          mlirDialectHandleLoadDialect(handle, context);
        }
      },
      py::arg("context") = py::none(), py::arg("load") = true);
}

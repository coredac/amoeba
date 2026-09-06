include(AddMLIRPython)

# Build a relocatable MLIR Python package containing Taskflow and Neura.
add_compile_definitions(
  "MLIR_PYTHON_PACKAGE_PREFIX=taskflow_mlir."
)

declare_mlir_python_sources(AmoebaPythonSources)

declare_mlir_dialect_python_bindings(
  ADD_TO_PARENT AmoebaPythonSources
  ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}"
  TD_FILE dialects/TaskflowOps.td
  SOURCES
    dialects/taskflow.py
  DIALECT_NAME taskflow
)

declare_mlir_python_extension(
  AmoebaPythonSources.PybindExtension

  MODULE_NAME _TaskflowExtensionPybind11
  ADD_TO_PARENT AmoebaPythonSources

  SOURCES
    TaskflowExtensionPybind11.cpp

  EMBED_CAPI_LINK_LIBS
    MLIRTaskflowCAPI

  PYTHON_BINDINGS_LIBRARY pybind11
)

set(
  AMOEBA_PYTHON_PACKAGE_DIR
  "${AMOEBA_PYTHON_PACKAGE_ROOT}/taskflow_mlir"
)

# Keep the packaged Python sources and their embedded C API dependencies in
# one list.
set(AMOEBA_PYTHON_SOURCES
  MLIRPythonSources.Core
  MLIRPythonExtension.RegisterEverything

  MLIRPythonSources.Dialects.builtin
  MLIRPythonSources.Dialects.affine
  MLIRPythonSources.Dialects.arith
  MLIRPythonSources.Dialects.cf
  MLIRPythonSources.Dialects.complex
  MLIRPythonSources.Dialects.func
  MLIRPythonSources.Dialects.linalg
  MLIRPythonSources.Dialects.math
  MLIRPythonSources.Dialects.memref
  MLIRPythonSources.Dialects.scf
  MLIRPythonSources.Dialects.tensor
  MLIRPythonSources.Dialects.vector

  NeuraPythonSources
  AmoebaPythonSources
)

add_mlir_python_common_capi_library(AmoebaPythonCAPI
  INSTALL_COMPONENT AmoebaPythonModules

  INSTALL_DESTINATION
    python_packages/amoeba_core/taskflow_mlir/_mlir_libs

  OUTPUT_DIRECTORY
    "${AMOEBA_PYTHON_PACKAGE_DIR}/_mlir_libs"

  RELATIVE_INSTALL_ROOT "../../../.."

  DECLARED_SOURCES
    ${AMOEBA_PYTHON_SOURCES}
)

add_mlir_python_modules(AmoebaPythonModules
  ROOT_PREFIX
    "${AMOEBA_PYTHON_PACKAGE_DIR}"

  INSTALL_PREFIX
    "python_packages/amoeba_core/taskflow_mlir"

  DECLARED_SOURCES
    ${AMOEBA_PYTHON_SOURCES}

  COMMON_CAPI_LINK_LIBS
    AmoebaPythonCAPI
)

# Neura's generated Python source is produced in a sibling CMake directory.
# Make the cross-directory generation order explicit before Amoeba copies it.
add_dependencies(
  AmoebaPythonModules.sources.NeuraPythonSources.neura.ops_gen
  NeuraPythonSources.neura.tablegen
)

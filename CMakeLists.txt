cmake_minimum_required(VERSION 3.13.4)

if(NOT DEFINED CMAKE_TOOLCHAIN_FILE)
  if(NOT DEFINED CMAKE_C_COMPILER AND NOT DEFINED ENV{CC})
    set(CMAKE_C_COMPILER clang CACHE FILEPATH "Default C compiler")
  endif()
  if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX})
    set(CMAKE_CXX_COMPILER clang++ CACHE FILEPATH "Default C++ compiler")
  endif()
endif()

project(AMOEBAMLIRProject VERSION 0.1.0)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  # The pinned MLIR revision triggers a known false positive when a
  # captureless lambda is assigned to llvm::function_ref.
  # See https://github.com/llvm/llvm-project/issues/126600.
  # Remove this workaround after updating LLVM past the upstream fix.
  add_compile_options(-Wno-dangling-assignment-gsl)
endif()

# Avoid cmake generate unknown warnings
cmake_policy(SET CMP0116 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_compile_options(-g)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  if(NOT DEFINED ENV{LLVM_BUILD_DIR})
    message(FATAL_ERROR
      "LLVM_BUILD_DIR is not set. "
      "Set it to /path/to/llvm-project/build."
    )
  endif()

  set(
    AMOEBA_LLVM_BUILD_DIR
    "$ENV{LLVM_BUILD_DIR}"
    CACHE PATH
    "LLVM build directory used by Amoeba"
  )

  set(
    LLVM_DIR
    "${AMOEBA_LLVM_BUILD_DIR}/lib/cmake/llvm"
    CACHE PATH
    "LLVM CMake package directory"
  )

  set(
    MLIR_DIR
    "${AMOEBA_LLVM_BUILD_DIR}/lib/cmake/mlir"
    CACHE PATH
    "MLIR CMake package directory"
  )

  if(DEFINED ENV{CONDA_PREFIX})
    set(
      Python3_EXECUTABLE
      "$ENV{CONDA_PREFIX}/bin/python"
      CACHE FILEPATH
      "Python3 interpreter used by Amoeba"
    )

    set(
      Python_EXECUTABLE
      "$ENV{CONDA_PREFIX}/bin/python"
      CACHE FILEPATH
      "Python interpreter used by Amoeba"
    )

    set(
      Python3_ROOT_DIR
      "$ENV{CONDA_PREFIX}"
      CACHE PATH
      "Python3 environment root"
    )

    set(
      Python_ROOT_DIR
      "$ENV{CONDA_PREFIX}"
      CACHE PATH
      "Python environment root"
    )
  endif()
endif()

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
# Configure the same Python development dependencies used by upstream MLIR.
include(MLIRDetectPythonEnv)
mlir_configure_python_dev_packages()

if(NOT TARGET MLIRPythonSources.Core)
  message(FATAL_ERROR
    "Amoeba requires MLIR Python bindings. "
    "Rebuild LLVM/MLIR with "
    "-DMLIR_ENABLE_BINDINGS_PYTHON=ON."
  )
endif()

set(
  AMOEBA_PYTHON_PACKAGE_ROOT
  "${CMAKE_BINARY_DIR}/python_packages/amoeba_core"
)

include_directories(${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

# Neura backend headers provided by the submodule.
include_directories(
  ${PROJECT_SOURCE_DIR}/thirdparty/neura/include
  ${PROJECT_BINARY_DIR}/thirdparty/neura/include
)

add_subdirectory(thirdparty)
add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(python)

# ===--------------------------------------------------------------------===#
# LIT test setup
# ===--------------------------------------------------------------------===#

# Tools built by this project or LLVM
set(MLIR_AMOEBA_OPT ${CMAKE_BINARY_DIR}/tools/mlir-amoeba-opt/mlir-amoeba-opt)
set(AMOEBA_COMPILER ${CMAKE_BINARY_DIR}/tools/amoeba-compiler/amoeba-compiler)
set(FILECHECK ${LLVM_TOOLS_BINARY_DIR}/FileCheck)
set(MLIR_OPT ${LLVM_TOOLS_BINARY_DIR}/mlir-opt)
set(MLIR_TRANSLATE ${LLVM_TOOLS_BINARY_DIR}/mlir-translate)
set(LLC ${LLVM_TOOLS_BINARY_DIR}/llc)
set(CLANG ${LLVM_TOOLS_BINARY_DIR}/clang)

# Configures lit.cfg from lit.cfg.in
configure_file(
  ${CMAKE_SOURCE_DIR}/test/lit.cfg.in
  ${CMAKE_SOURCE_DIR}/test/lit.cfg
  @ONLY
)

if(NOT LLVM_EXTERNAL_LIT)
  set(LLVM_EXTERNAL_LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
endif()

# Adds a custom target for running lit tests
add_custom_target(check-amoeba
  COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_SOURCE_DIR}/test
  DEPENDS
    mlir-amoeba-opt
    AmoebaPythonModules
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running Amoeba Dialect Tests with lit"
)
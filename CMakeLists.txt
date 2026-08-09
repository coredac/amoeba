cmake_minimum_required(VERSION 3.13.4)
project(AMOEBAMLIRProject VERSION 0.1.0)

# Avoid cmake generate unknown warnings
cmake_policy(SET CMP0116 NEW)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

add_compile_options(-g)

message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")

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
   ${CMAKE_BINARY_DIR}/test/lit.cfg
  # ${CMAKE_SOURCE_DIR}/test/lit.cfg
  @ONLY
)

if(NOT LLVM_EXTERNAL_LIT)
  set(LLVM_EXTERNAL_LIT "${LLVM_TOOLS_BINARY_DIR}/llvm-lit")
endif()

# Adds a custom target for running lit tests
add_custom_target(check-amoeba
  COMMAND ${LLVM_EXTERNAL_LIT} -sv ${CMAKE_BINARY_DIR}/test
  DEPENDS mlir-amoeba-opt
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  COMMENT "Running Amoeba Dialect Tests with lit"
)
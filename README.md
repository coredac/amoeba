# Amoeba

Amoeba is an MLIR project for representing and transforming portable
task-level dataflow programs. It owns the Taskflow dialect, generic Taskflow
transformations, and the integration layer used to connect Taskflow programs
to architecture-specific backends.

[Neura](https://github.com/coredac/neura) is currently the first supported
backend and is consumed as a pinned Git submodule. The project is structured
so that additional spatial dataflow backends can be integrated without moving
architecture-specific concepts into the Taskflow core.

## Architecture

```text
Frontend and loop dialects
          |
          v
  Taskflow dialect             portable task abstraction
          |
          v
  Backend integration          backend-specific conversion and optimization
          |
          v
  Neura backend                current multi-CGRA backend
```

The ownership boundary is:

- **Amoeba core** owns the Taskflow dialect and backend-independent passes.
- **Amoeba backend adapters** translate and optimize Taskflow for a particular
  backend.
- **Backend projects** own their target dialects, architecture models,
  lowerings, mapping, and target-specific transformations.

See the [Backend integration guide](include/Backend/README.md) for the detailed
layout, the Neura integration path, and instructions for adding another
backend.

## Repository layout

```text
amoeba/
|-- include/
|   |-- Taskflow-C/            # Public Taskflow C API used by Python bindings
|   |-- TaskflowDialect/       # Taskflow dialect and pass interfaces
|   |-- Conversion/            # Backend-independent conversion interfaces
|   `-- Backend/               # Public backend integration interfaces
|-- lib/
|   |-- CAPI/                  # Taskflow C API implementation
|   |-- TaskflowDialect/       # Taskflow implementation and generic passes
|   |-- Conversion/            # Backend-independent conversions
|   `-- Backend/               # Backend adapter implementations
|-- python/
|   `-- dialects/              # Unified Taskflow and Neura Python package
|-- thirdparty/
|   `-- neura/                 # Neura Git submodule
|-- tools/
|   `-- mlir-amoeba-opt/       # Amoeba optimizer driver
`-- test/                      # Core, conversion, end-to-end, and backend tests
```

## Prerequisites

Amoeba requires:

- a C++17 compiler, with `clang` and `clang++` used by the reference build;
- CMake, Ninja, and Make;
- `ccache` and `lld` for the reference LLVM build;
- Python 3.11;
- LLVM/MLIR at commit
  [`6146a88f60492b520a36f8f8f3231e15f3cc6082`](https://github.com/llvm/llvm-project/commit/6146a88f60492b520a36f8f8f3231e15f3cc6082);
  and
- Git submodule support.

Install the Python build dependencies into the same environment used to
configure LLVM and Amoeba:

```bash
python -m pip install pybind11==2.13.6 nanobind==2.15.0
```

This is the same LLVM revision used by the current
[Neura build instructions](https://github.com/coredac/neura#build-llvm--neura)
and [Amoeba CI](.github/workflows/main.yml).

## Checkout

Clone Amoeba together with its backend dependency:

```bash
git clone --recurse-submodules git@github.com:coredac/amoeba.git
cd amoeba
```

For an existing checkout, initialize or update the pinned Neura revision with:

```bash
git submodule update --init --recursive
git submodule status
```

## Build LLVM and MLIR

Clone LLVM, check out the pinned revision, and create an out-of-tree build:

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout 6146a88f60492b520a36f8f8f3231e15f3cc6082
mkdir build && cd build

cmake -G Ninja ../llvm \
  -DLLVM_ENABLE_PROJECTS="mlir;clang" \
  -DLLVM_BUILD_EXAMPLES=OFF \
  -DLLVM_TARGETS_TO_BUILD="Native" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_FLAGS="-std=c++17 -frtti" \
  -DLLVM_ENABLE_LLD=ON \
  -DMLIR_INSTALL_AGGREGATE_OBJECTS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DMLIR_BINDINGS_PYTHON_NB_DOMAIN=mlir \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)" \
  -DLLVM_CCACHE_BUILD=ON \
  -DCMAKE_C_COMPILER_LAUNCHER=ccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=ccache

cmake --build . --parallel 2
```

The LLVM and MLIR test suites can be checked with:

```bash
cmake --build . --target check-mlir
cmake --build . --target check-clang
```

## Build Amoeba

From the Amoeba repository root, set `LLVM_BUILD_DIR` to the LLVM build above.
Amoeba derives the LLVM source and CMake package paths from this directory:

```bash
export LLVM_BUILD_DIR=/path/to/llvm-project/build

cmake -G Ninja -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython3_EXECUTABLE="$(which python)" \
  -DPython_EXECUTABLE="$(which python)"

cmake --build build --parallel 2
cmake --build build --target AmoebaPythonModules --parallel 2
```

Amoeba assembles Taskflow and Neura into one MLIR Python package at
`build/python_packages/amoeba_core/taskflow_mlir`. Both dialects therefore share
the same Python `Context`, `Type`, and native MLIR runtime.

The Neura submodule remains `EXCLUDE_FROM_ALL`: Amoeba reuses the Neura
libraries and Python sources it needs without building standalone Neura tools.

## Command-line tool

The build produces `mlir-amoeba-opt`, an `mlir-opt`-style driver that registers
the Taskflow dialect, Amoeba conversions, and all backends enabled in the
build:

```bash
./build/tools/mlir-amoeba-opt/mlir-amoeba-opt --help
```

For example, an affine program can be converted to Taskflow with:

```bash
./build/tools/mlir-amoeba-opt/mlir-amoeba-opt input.mlir \
  --convert-affine-to-taskflow \
  -o output.mlir
```

Neura-specific passes and options are registered by the Neura adapter. The
architecture and latency options are exposed as `--neura-architecture-spec`
and `--neura-latency-spec`.

## Tests

After building Amoeba, run the focused Python binding test:

```bash
$LLVM_BUILD_DIR/bin/llvm-lit -v \
  --filter='taskflow_neura_binding.py' test
```

Run the complete suite through the CMake target:

```bash
cmake --build build --target check-amoeba
```

The suite covers Taskflow conversions and transformations, the unified
Taskflow/Neura Python package, end-to-end lowering, and Neura backend
integration.

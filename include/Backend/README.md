# Amoeba backend integration

Amoeba separates its portable Taskflow abstraction from the details of any
particular target. A backend adapter is the boundary between those layers: it
registers a target dialect and its passes, and owns the Taskflow conversions,
orchestration, and optimization that are meaningful only for that target.

This README lives next to the public backend interfaces under
`include/Backend`, but it documents both the public interfaces and their
implementations under `lib/Backend`. Keeping one backend guide avoids
duplicating documentation across the two trees.

## Directory overview

```text
include/Backend/
|-- Backends.h                 # Common registration entry point
`-- Neura/
    |-- NeuraBackend.h         # Public Neura registration interface
    |-- NeuraBackendPasses.h   # Generated-pass declarations
    |-- NeuraBackendPasses.td  # Neura adapter pass definitions
    `-- Orchestration/         # Public orchestration interfaces

lib/Backend/
|-- Backends.cpp               # Registers every enabled backend
`-- Neura/
    |-- NeuraBackend.cpp       # Neura options and registration
    |-- Conversion/            # Taskflow-to-Neura conversion
    |-- Orchestration/         # Multi-CGRA placement and scheduling
    `-- Transforms/            # Neura-specific Taskflow transforms

thirdparty/
`-- neura/                     # Pinned Neura Git submodule
```

Public declarations belong under `include/Backend/<Backend>`. Implementations
and private backend organization belong under `lib/Backend/<Backend>`. The
target implementation itself remains in its own project or dependency.

## Ownership boundary

| Layer | Owns | Current examples |
| --- | --- | --- |
| Amoeba core | Portable Taskflow IR and backend-independent transformations | Taskflow dialect, affine-to-Taskflow conversion, generic affine transforms |
| Amoeba backend adapter | Translation and optimization decisions specific to one backend | Taskflow-to-Neura conversion, hyperblock construction, task classification, orchestration, fusion, resource-aware optimization |
| Backend project | Target IR, architecture model, lowerings, mapping, and target transforms | Neura dialect, architecture parser, Neura lowering and mapping passes |

The `include/Backend/Neura` and `lib/Backend/Neura` directories are therefore
not copies of the Neura project. They are the Amoeba-owned adapter between
Taskflow and the Neura API supplied by the submodule.

## How Neura is integrated

The current integration follows this path:

1. [`.gitmodules`](../../.gitmodules) declares `thirdparty/neura` as a Git
   submodule of `coredac/neura`. The gitlink pins the exact Neura revision used
   by an Amoeba commit.

2. [`thirdparty/CMakeLists.txt`](../../thirdparty/CMakeLists.txt) adds the
   submodule to the build:

   ```cmake
   add_subdirectory(neura EXCLUDE_FROM_ALL)
   ```

3. The [top-level CMake configuration](../../CMakeLists.txt) exposes the Neura
   source and generated include directories, then adds `thirdparty` before the
   Amoeba libraries and tools.

4. [`lib/Backend/Neura/CMakeLists.txt`](../../lib/Backend/Neura/CMakeLists.txt)
   builds the Amoeba-side adapter and links it to targets supplied by the
   submodule, including:

   ```text
   MLIRNeura
   MLIRNeuraTransforms
   MLIRNeuraConversion
   MLIRNUERAOptimization
   ```

5. [`registerNeuraBackend()`](../../lib/Backend/Neura/NeuraBackend.cpp)
   registers both sides of the integration:

   ```cpp
   void mlir::amoeba::registerNeuraBackend(DialectRegistry &registry) {
     registry.insert<mlir::neura::NeuraDialect>();
     mlir::neura::registerPasses();
     mlir::registerNeuraConversionPasses();
     mlir::amoeba::neura::registerNeuraBackendPasses();
     mlir::amoeba::neura::registerTaskflowConversionPassPipeline();
   }
   ```

   The `mlir::neura` calls refer to APIs provided by the submodule. The
   `mlir::amoeba::neura` calls register adapter passes owned by Amoeba.

6. [`registerBackends()`](../../lib/Backend/Backends.cpp) is the common entry
   point. It currently calls `registerNeuraBackend()`.

7. [`mlir-amoeba-opt`](../../tools/mlir-amoeba-opt/mlir-amoeba-opt.cpp) calls
   `registerBackends()` when constructing its dialect registry. The CLI remains
   backend-neutral while each adapter owns its target-specific registration
   and command-line options.

The complete flow is:

```text
.gitmodules
    |
    v
thirdparty/neura CMake targets
    |
    v
MLIRAmoebaNeuraBackend
    |
    v
registerNeuraBackend()
    |
    v
registerBackends()
    |
    v
mlir-amoeba-opt
```

## Adding another backend

A new backend should follow the same boundary:

1. **Provide the target dependency.** Add a submodule under `thirdparty`, use
   `find_package`, or link another externally provided CMake target. A Git
   submodule is an integration choice, not a requirement for every backend.

2. **Define the public adapter interface.** Add
   `include/Backend/<Backend>/<Backend>Backend.h` with a registration function:

   ```cpp
   void registerExampleBackend(mlir::DialectRegistry &registry);
   ```

3. **Implement the adapter.** Add `lib/Backend/<Backend>` for backend-specific
   conversions, transforms, pipelines, orchestration, and options. Keep generic
   Taskflow transformations in the Amoeba core.

4. **Create one adapter library target.** Link its internal components and the
   target project libraries behind a backend-level CMake target, following
   `MLIRAmoebaNeuraBackend`.

5. **Register the backend.** Call the new registration function from
   [`Backends.cpp`](../../lib/Backend/Backends.cpp) and link its adapter target
   from [`lib/Backend/CMakeLists.txt`](../../lib/Backend/CMakeLists.txt).

6. **Add integration tests.** Test the Taskflow-to-backend boundary and the
   backend pipeline without weakening the backend-independent Taskflow tests.

Architecture files, latency models, and target-specific command-line options
should stay in the adapter or backend project. They should not be added to the
generic `mlir-amoeba-opt` driver or the Taskflow core.

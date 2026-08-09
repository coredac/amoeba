// AmoebaConversionPasses.td - Passes for Amoeba dialect conversion

#ifndef AMOEBA_CONVERSION_PASSES_TD
#define AMOEBA_CONVERSION_PASSES_TD

include "mlir/Pass/PassBase.td"

//=========================================================//
// Taskflow conversion passes.
//=========================================================//

def ConvertAffineToTaskflow : Pass<"convert-affine-to-taskflow", "ModuleOp"> {
  let summary = "Convert top-level affine.for operations to Taskflow dialect";
  let description = [{
    This pass converts top-level affine.for loops in a function into
    taskflow.task operations within a taskflow.graph. Each top-level loop
    becomes a separate task, and data dependencies between tasks are made
    explicit through taskflow.channel operations.

    The pass:
    1. Identifies all top-level affine.for operations
    2. Analyzes data dependencies (RAW, WAR, WAW) between loops
    3. Creates a taskflow.graph containing the loops
    4. Converts each loop to a taskflow.task
    5. Inserts taskflow.channel operations for data dependencies
  }];

  let constructor = "mlir::createConvertAffineToTaskflowPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::affine::AffineDialect",
                           "mlir::func::FuncDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect"];
}

def ConvertTaskflowToNeura : Pass<"convert-taskflow-to-neura", "ModuleOp"> {
  let summary = "Convert taskflow operations to neura.kernel";
  let description = [{
    Converts taskflow.hyperblock operations into neura.kernel operations suitable for
    CGRA tile array mapping.

    Converts the entire taskflow.hyperblock operations and corresponding counters ops
    to neura.kernel, based on the input architecture spec file.

    mlir-amoeba-opt --convert-taskflow-to-neura input.mlir --architecture-spec=$PATH/architecture.yaml -o output.mlir
  }];
  let constructor = "mlir::createConvertTaskflowToNeuraPass()";
  let dependentDialects = ["mlir::taskflow::TaskflowDialect",
                           "mlir::neura::NeuraDialect",
                           "mlir::affine::AffineDialect"];
}

def FoldSubView : Pass<"fold-subview", "func::FuncOp"> {
  let summary = "Folds memref.subview into affine load/store operations";
  let description = [{
    Eliminates memref.subview operations by folding them into their users
    (affine.load and affine.store). Adjusts indices to account for subview
    offsets and strides, enabling direct access to the source memref.
  }];
  let constructor = "mlir::createFoldSubViewPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::func::FuncDialect"];
}

def ConvertCopyToAffineLoops
    : Pass<"convert-copy-to-affine-loops", "func::FuncOp"> {
  let summary = "Converts memref.copy to explicit affine loop nests";
  let description = [{
    Converts memref.copy operations into explicit affine.for loop nests
    with affine.load and affine.store operations. This makes data movement
    explicit for CGRA compilation with global addressing.
  }];
  let constructor = "mlir::createConvertCopyToAffineLoopsPass()";
  let dependentDialects = ["mlir::affine::AffineDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::func::FuncDialect"];
}

#endif // AMOEBA_CONVERSION_PASSES_TD

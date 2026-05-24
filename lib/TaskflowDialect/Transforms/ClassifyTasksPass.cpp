//===- ClassifyTasksPass.cpp - Classify taskflow tasks by bound type ------===//
//
// Classifies each taskflow.task as one of three categories and stores the
// result in a "task_type" string attribute on the op:
//
//   "static"            – all counter bounds are compile-time constants.
//   "symbol_dynamic"    – bounds are runtime values that are fixed for the
//                         entire function invocation (e.g. from memref.dim of
//                         a function argument), so the iteration count is
//                         known before the task is launched.
//   "irregular_dynamic" – bounds depend on data produced by earlier tasks;
//                         the iteration count cannot be determined until those
//                         tasks have completed.
//
// Classification per counter bound uses the following rules:
//   1. arith.constant inside the task body              → static
//   2. block arg (value_input) whose original value is:
//        arith.constant at call site                    → static
//        function argument (any type)                   → symbol_dynamic
//        memref.dim                                     → symbol_dynamic
//   3. any other op computed inside the task body       → irregular_dynamic
//
// The most-dynamic category across ALL counters in a task determines the
// task's category ("irregular" beats "symbol", which beats "static").
//
//===----------------------------------------------------------------------===//

#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"

#include <memory>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Bound classification helpers
//===----------------------------------------------------------------------===//

// Bound categories ordered from least to most dynamic.
enum class BoundKind { Static = 0, SymbolDynamic = 1, IrregularDynamic = 2 };

static BoundKind worstCase(BoundKind a, BoundKind b) {
  return static_cast<BoundKind>(
      std::max(static_cast<int>(a), static_cast<int>(b)));
}

// Classifies a value that lives OUTSIDE the task body (i.e. a value_input
// operand at the call site) by tracing it to its origin.
static BoundKind classifyOuterValue(Value v) {
  // Function argument of any type → symbol_dynamic (fixed per invocation).
  if (auto block_arg = dyn_cast<BlockArgument>(v)) {
    auto *parent_region = block_arg.getParentRegion();
    if (parent_region && isa<func::FuncOp>(parent_region->getParentOp())) {
      return BoundKind::SymbolDynamic;
    }
    // Block arg in some other region (e.g. affine.for) — conservative.
    return BoundKind::IrregularDynamic;
  }

  Operation *def = v.getDefiningOp();
  if (!def) {
    return BoundKind::IrregularDynamic;
  }

  // Compile-time constant → static.
  if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
    return BoundKind::Static;
  }

  // memref.dim — classification depends on what the memref is.
  if (auto dim_op = dyn_cast<memref::DimOp>(def)) {
    // Shapes are fixed at allocation time and no task can change them, so the
    // result of memref.dim is always known before any task that uses it as a
    // value_input is launched → symbol_dynamic.
    return BoundKind::SymbolDynamic;
  }

  // Any other computation is considered irregular.
  return BoundKind::IrregularDynamic;
}

// Classifies a single bound value as seen INSIDE the task body.
// task_op is used to map block arguments back to their value_input operands.
static BoundKind classifyTaskBoundValue(Value v, TaskflowTaskOp task_op) {
  // Op defined inside the task body.
  if (Operation *def = v.getDefiningOp()) {
    if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
      return BoundKind::Static;
    }
    // Any other computed value is data-dependent.
    return BoundKind::IrregularDynamic;
  }

  // Block argument of the task body.
  auto block_arg = dyn_cast<BlockArgument>(v);
  if (!block_arg) {
    return BoundKind::IrregularDynamic;
  }

  // Block args layout:  [dep_read_in…] [dep_write_in…] [value_inputs…]
  unsigned num_dep_read = task_op.getDependencyReadIn().size();
  unsigned num_dep_write = task_op.getDependencyWriteIn().size();
  unsigned value_input_start = num_dep_read + num_dep_write;
  unsigned arg_idx = block_arg.getArgNumber();

  // A taskflow.counter bound is always of index type and can therefore
  // never be a memref dependency block argument.
  assert(arg_idx >= value_input_start &&
         "Counter bounds should not be memref dependency block args");

  unsigned vi_idx = arg_idx - value_input_start;
  auto value_inputs = task_op.getValueInputs();
  assert(vi_idx < value_inputs.size() &&
         "Counter bound block argument index out of range of value inputs");

  return classifyOuterValue(value_inputs[vi_idx]);
}

//===----------------------------------------------------------------------===//
// Per-task classification
//===----------------------------------------------------------------------===//

static LogicalResult classifyTask(TaskflowTaskOp task_op) {
  // Detects if construct-hyperblock-from-task has not run yet.
  bool has_affine_for = false;
  task_op.walk([&](affine::AffineForOp) -> WalkResult {
    has_affine_for = true;
    return WalkResult::interrupt();
  });
  bool has_counter = false;
  task_op.walk([&](TaskflowCounterOp) -> WalkResult {
    has_counter = true;
    return WalkResult::interrupt();
  });
  if (has_affine_for && !has_counter) {
    return task_op.emitError()
           << "[ClassifyTasks]: task '" << task_op.getTaskName()
           << "' contains affine.for loops but no taskflow.counter ops — "
              "run 'construct-hyperblock-from-task' before 'classify-tasks'";
  }

  BoundKind task_kind = BoundKind::Static;

  task_op.walk([&](TaskflowCounterOp counter_op) {
    // Inspect lower bound, upper bound, and step.
    for (Value bound : {counter_op.getLowerBound(), counter_op.getUpperBound(),
                        counter_op.getStep()}) {
      BoundKind bk = classifyTaskBoundValue(bound, task_op);
      task_kind = worstCase(task_kind, bk);
    }
  });

  StringRef type_str;
  switch (task_kind) {
  case BoundKind::Static:
    type_str = "static";
    break;
  case BoundKind::SymbolDynamic:
    type_str = "symbol_dynamic";
    break;
  case BoundKind::IrregularDynamic:
    type_str = "irregular_dynamic";
    break;
  }

  OpBuilder builder(task_op.getContext());
  task_op->setAttr("task_type", builder.getStringAttr(type_str));
  return success();
}

//===----------------------------------------------------------------------===//
// Pass definition
//===----------------------------------------------------------------------===//

struct ClassifyTasksPass
    : public PassWrapper<ClassifyTasksPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClassifyTasksPass)

  StringRef getArgument() const override { return "classify-tasks"; }
  StringRef getDescription() const override {
    return "Classify taskflow.task ops as static, symbol_dynamic, or "
           "irregular_dynamic based on counter bound origins.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult result = module.walk([&](TaskflowTaskOp task_op) -> WalkResult {
      if (failed(classifyTask(task_op))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::taskflow::createClassifyTasksPass() {
  return std::make_unique<ClassifyTasksPass>();
}

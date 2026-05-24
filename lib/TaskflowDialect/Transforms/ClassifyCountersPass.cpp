#include "TaskflowDialect/TaskflowDialect.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "TaskflowDialect/TaskflowPasses.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/TypeID.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include <memory>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

//===----------------------------------------------------------------------===//
// Bound kind classification
//===----------------------------------------------------------------------===//

enum class BoundKind { Static = 0, SymbolDynamic = 1, IrregularDynamic = 2 };

static BoundKind worstCase(BoundKind a, BoundKind b) {
  return static_cast<BoundKind>(std::max((int)a, (int)b));
}

static StringRef boundKindToStr(BoundKind k) {
  switch (k) {
  case BoundKind::Static:
    return "static";
  case BoundKind::SymbolDynamic:
    return "symbol_dynamic";
  case BoundKind::IrregularDynamic:
    return "irregular_dynamic";
  }
  llvm_unreachable("unknown BoundKind");
}

// Classifies a value that lives outside any taskflow.task region.
static BoundKind classifyOuterValue(Value v) {
  if (auto block_arg = dyn_cast<BlockArgument>(v)) {
    auto *parent_region = block_arg.getParentRegion();
    if (parent_region && isa<func::FuncOp>(parent_region->getParentOp())) {
      return BoundKind::SymbolDynamic;
    }
    return BoundKind::IrregularDynamic;
  }
  Operation *def = v.getDefiningOp();
  if (!def) {
    return BoundKind::IrregularDynamic;
  }
  if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
    return BoundKind::Static;
  }
  if (isa<memref::DimOp>(def)) {
    return BoundKind::SymbolDynamic;
  }
  return BoundKind::IrregularDynamic;
}

// Classifies a bound value inside a taskflow.task body.
static BoundKind classifyTaskBoundValue(Value v, TaskflowTaskOp task_op) {
  if (Operation *def = v.getDefiningOp()) {
    if (isa<arith::ConstantOp, arith::ConstantIndexOp>(def)) {
      return BoundKind::Static;
    }
    // affine.apply is a pure affine computation; its dynamism is fully
    // determined by its operands (e.g. affine_map<()[s0]->(s0-2)>()[%n]).
    if (isa<affine::AffineApplyOp>(def)) {
      BoundKind k = BoundKind::Static;
      for (Value operand : def->getOperands()) {
        k = worstCase(k, classifyTaskBoundValue(operand, task_op));
      }
      return k;
    }
    return BoundKind::IrregularDynamic;
  }
  auto block_arg = dyn_cast<BlockArgument>(v);
  if (!block_arg) {
    return BoundKind::IrregularDynamic;
  }

  unsigned num_dep_read = task_op.getDependencyReadIn().size();
  unsigned num_dep_write = task_op.getDependencyWriteIn().size();
  unsigned value_input_start = num_dep_read + num_dep_write;
  unsigned arg_idx = block_arg.getArgNumber();

  assert(arg_idx >= value_input_start &&
         "counter bound is a memref dependency block arg — IR is malformed");

  unsigned vi_idx = arg_idx - value_input_start;
  auto value_inputs = task_op.getValueInputs();
  assert(vi_idx < value_inputs.size() &&
         "counter bound block arg index out of range of value inputs");

  return classifyOuterValue(value_inputs[vi_idx]);
}

// Returns the worst-case BoundKind across lb/ub/step of a single counter.
static BoundKind classifyCounterBound(TaskflowCounterOp counter_op,
                                      TaskflowTaskOp task_op) {
  BoundKind k = BoundKind::Static;
  for (Value bound : {counter_op.getLowerBound(), counter_op.getUpperBound(),
                      counter_op.getStep()}) {
    k = worstCase(k, classifyTaskBoundValue(bound, task_op));
  }
  return k;
}

//===----------------------------------------------------------------------===//
// Counter structural + bound classification
//===----------------------------------------------------------------------===//

void classifyCountersInTask(TaskflowTaskOp task_op) {
  // Collects all counters in the task.
  SmallVector<TaskflowCounterOp> counters;
  task_op.walk(
      [&](TaskflowCounterOp counter_op) { counters.push_back(counter_op); });

  if (counters.empty()) {
    return;
  }

  // Builds parent-child relationships.
  // Maps from counter results to counter ops.
  DenseMap<Value, TaskflowCounterOp> value_to_counter;
  for (TaskflowCounterOp counter_op : counters) {
    value_to_counter[counter_op.getCounterIndex()] = counter_op;
  }

  // Finds which counters have children.
  DenseSet<TaskflowCounterOp> counters_with_children;
  for (TaskflowCounterOp counter_op : counters) {
    if (auto parent_idx = counter_op.getParentIndex()) {
      if (auto parent_counter = value_to_counter.lookup(parent_idx)) {
        counters_with_children.insert(parent_counter);
      }
    }
  }

  int global_counter_id = 0;

  // Classifies each counter and sets counter_type = [structural, bound_kind].
  OpBuilder builder(task_op.getContext());
  for (TaskflowCounterOp counter_op : counters) {
    bool has_parent = (counter_op.getParentIndex() != nullptr);
    bool has_child = counters_with_children.contains(counter_op);

    StringRef structural_type;
    if (!has_parent && !has_child) {
      // Single loop: treat as leaf counter (can be mapped to the CGRA tile
      // array).
      structural_type = "leaf";
    } else if (!has_parent && has_child) {
      // Root counter: top-level loop with nested loops.
      structural_type = "root";
    } else if (has_parent && has_child) {
      // Relay counter: nested loop with further nested loops.
      structural_type = "relay";
    } else {
      // Leaf counter: innermost loop.
      structural_type = "leaf";
    }

    StringRef bound_type =
        boundKindToStr(classifyCounterBound(counter_op, task_op));

    // Sets counter_hierarchy (structural role) and counter_dynamism (bound
    // kind).
    counter_op.setCounterHierarchyAttr(builder.getStringAttr(structural_type));
    counter_op.setCounterDynamismAttr(builder.getStringAttr(bound_type));
    // Sets the counter id attribute.
    counter_op.setCounterIdAttr(builder.getI32IntegerAttr(global_counter_id++));
  }
}

struct ClassifyCountersPass
    : public PassWrapper<ClassifyCountersPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ClassifyCountersPass)

  StringRef getArgument() const override { return "classify-counters"; }
  StringRef getDescription() const override {
    return "Classify taskflow counters as root/relay/leaf.";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    module.walk(
        [&](TaskflowTaskOp task_op) { classifyCountersInTask(task_op); });
  }
};
} // namespace

std::unique_ptr<Pass> mlir::taskflow::createClassifyCountersPass() {
  return std::make_unique<ClassifyCountersPass>();
}
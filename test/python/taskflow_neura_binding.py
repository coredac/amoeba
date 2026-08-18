# RUN: %python %s | FileCheck %s
# RUN: %python %s | mlir-amoeba-opt --leverage-predicated-value | FileCheck %s --check-prefix=PREDICATED

from taskflow_mlir.dialects import func, neura, taskflow
from taskflow_mlir.ir import (
    Context,
    DictAttr,
    InsertionPoint,
    IntegerAttr,
    IntegerType,
    Location,
    Module,
    StringAttr,
)


def build_placement(x, y, i32):
    return DictAttr.get(
        {
            "x": IntegerAttr.get(i32, x),
            "y": IntegerAttr.get(i32, y),
        }
    )


def build_module():
    with Context(), Location.unknown():
        taskflow.register_dialect()
        neura.register_dialect()

        module = Module.create()
        i32 = IntegerType.get_signless(32)

        with InsertionPoint(module.body):
            function = func.FuncOp("single_task", ([], []))
            entry_block = function.add_entry_block()

        with InsertionPoint(entry_block):
            task = taskflow.TaskflowTaskOp(
                [],
                [],
                [],
                [],
                [],
                [],
                "add_constant",
                [],
                [],
            )
            task_block = task.body.blocks.append()
            func.ReturnOp([])

        with InsertionPoint(task_block):
            kernel = neura.KernelOp(
                [],
                [],
                [],
                accelerator=StringAttr.get("neura"),
            )
            kernel_block = kernel.body.blocks.append()
            taskflow.TaskflowYieldOp([], [], [])

        with InsertionPoint(kernel_block):
            lhs = neura.ConstantOp(
                i32,
                IntegerAttr.get(i32, 1),
            )
            lhs.operation.attributes["placement"] = build_placement(0, 0, i32)

            rhs = neura.ConstantOp(
                i32,
                IntegerAttr.get(i32, 2),
            )
            rhs.operation.attributes["placement"] = build_placement(2, 0, i32)

            result = neura.AddOp(
                i32,
                lhs.result,
                rhs=rhs.result,
            )
            result.operation.attributes["placement"] = build_placement(1, 0, i32)

            neura.YieldOp([], [])

        assert module.operation.verify()
        return module


# CHECK:     module {
# CHECK-NEXT:  func.func @single_task() {
# CHECK-NEXT:    taskflow.task @add_constant : () -> () {
# CHECK-NEXT:      neura.kernel attributes {accelerator = "neura"} {
# CHECK-NEXT:        %0 = "neura.constant"() <{value = 1 : i32}> {placement = {x = 0 : i32, y = 0 : i32}} : () -> i32
# CHECK-NEXT:        %1 = "neura.constant"() <{value = 2 : i32}> {placement = {x = 2 : i32, y = 0 : i32}} : () -> i32
# CHECK-NEXT:        %2 = "neura.add"(%0, %1) {placement = {x = 1 : i32, y = 0 : i32}} : (i32, i32) -> i32
# CHECK-NEXT:        neura.yield
# CHECK-NEXT:      }
# CHECK-NEXT:      taskflow.yield
# CHECK-NEXT:    }
# CHECK-NEXT:    return
# CHECK-NEXT:  }
# CHECK-NEXT:}

# PREDICATED:     module {
# PREDICATED-NEXT:  func.func @single_task() {
# PREDICATED-NEXT:    taskflow.task @add_constant : () -> () {
# PREDICATED-NEXT:      neura.kernel attributes {accelerator = "neura"} {
# PREDICATED-NEXT:        %0 = "neura.constant"() <{value = 1 : i32}> {placement = {x = 0 : i32, y = 0 : i32}} : () -> !neura.data<i32, i1>
# PREDICATED-NEXT:        %1 = "neura.constant"() <{value = 2 : i32}> {placement = {x = 2 : i32, y = 0 : i32}} : () -> !neura.data<i32, i1>
# PREDICATED-NEXT:        %2 = "neura.add"(%0, %1) {placement = {x = 1 : i32, y = 0 : i32}} : (!neura.data<i32, i1>, !neura.data<i32, i1>) -> !neura.data<i32, i1>
# PREDICATED-NEXT:        neura.yield
# PREDICATED-NEXT:      }
# PREDICATED-NEXT:      taskflow.yield
# PREDICATED-NEXT:    }
# PREDICATED-NEXT:    return
# PREDICATED-NEXT:  }
# PREDICATED-NEXT:}

print(build_module())

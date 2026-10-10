// RUN: mlir-amoeba-opt %s \
// RUN:   --architecture-spec=%S/../archspec/architecture_with_counter.yaml \
// RUN:   --construct-hyperblock-from-task \
// RUN:   --classify-task-and-counter \
// RUN:   --convert-taskflow-to-neura \
// RUN:   | FileCheck %s

module {
  func.func @general(%output: memref<i32>, %condition: i1) {
    %done = taskflow.task @general
      will_writes(%output : memref<i32>)
      value_inputs(%condition : i1)
      [original_write_memrefs(%output : memref<i32>)]
      : (memref<i32>, i1) -> memref<i32> {
    ^bb0(%out: memref<i32>, %condition_arg: i1):
      scf.if %condition_arg {
        %one = arith.constant 1 : i32
        memref.store %one, %out[] : memref<i32>
        scf.yield
      } else {
        %zero = arith.constant 0 : i32
        memref.store %zero, %out[] : memref<i32>
        scf.yield
      }
      taskflow.yield done_writes(%out : memref<i32>)
    }
    return
  }
}

// CHECK: taskflow.task @general
// CHECK: neura.kernel
// CHECK: scf.if
// CHECK: memref.store
// CHECK: neura.yield
// CHECK-NOT: taskflow.hyperblock

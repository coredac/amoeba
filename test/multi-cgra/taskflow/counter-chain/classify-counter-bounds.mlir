// RUN: mlir-amoeba-opt %s --classify-task-and-counter > %t 2>&1
// RUN: FileCheck %s --input-file=%t

module {
  func.func @classify_counter_bounds(%symbol: index) {
    taskflow.task @Task_0 value_inputs(%symbol : index) : (index) -> () {
      ^bb0(%arg0: index):
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %constant = taskflow.counter from %c0 to %c2 step %c1 : index
        %symbolic = taskflow.counter from %c0 to %arg0 step %c1 : index
        %dynamic_upper = arith.addi %c2, %c3 : index
        %dynamic = taskflow.counter from %c0 to %dynamic_upper step %c1 : index
        taskflow.yield
    }
    return
  }
}

// CHECK-LABEL: module {
// CHECK-NEXT:   func.func @classify_counter_bounds(%arg0: index) {
// CHECK-NEXT:     taskflow.task @Task_0 value_inputs(%arg0 : index) : (index) -> () {
// CHECK-NEXT:       ^bb0(%arg1: index):
// CHECK-NEXT:         %c0 = arith.constant 0 : index
// CHECK-NEXT:         %c1 = arith.constant 1 : index
// CHECK-NEXT:         %c2 = arith.constant 2 : index
// CHECK-NEXT:         %c3 = arith.constant 3 : index
// CHECK-NEXT:         %0 = taskflow.counter from %c0 to %c2 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// CHECK-NEXT:         %1 = taskflow.counter from %c0 to %arg1 step %c1 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} : index
// CHECK-NEXT:         %2 = arith.addi %c2, %c3 : index
// CHECK-NEXT:         %3 = taskflow.counter from %c0 to %2 step %c1 attributes {counter_dynamism = "dynamic_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
// CHECK-NEXT:         taskflow.yield
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK-NEXT: }
// CHECK-NOT: {{.}}

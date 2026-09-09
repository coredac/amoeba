// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   --mlir-print-op-on-diagnostic=false > %t.diagnostic 2>&1; \
// RUN:   status=$?; FileCheck %s --input-file=%t.diagnostic; check_status=$?; \
// RUN:   test $status -ne 0 -a $check_status -eq 0

module {
  func.func @main(%upper: index, %input: i32) -> i32 {
    %result = taskflow.task @A value_inputs(%upper, %input : index, i32)
        : (index, i32) -> i32 {
    ^bb0(%task_upper: index, %task_input: i32):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %task_upper step %c1 : index
      taskflow.yield values(%task_input : i32)
    }
    return %result : i32
  }
}

// CHECK-LABEL: {{.*}}reject-non-static-counter.mlir:9:3: error: task A requires constant counter bounds, a positive step, a non-empty range, and a trip count within int64; static shape selection requires constant counter bounds
// CHECK-NEXT:   func.func @main(%upper: index, %input: i32) -> i32 {
// CHECK-NEXT:   ^

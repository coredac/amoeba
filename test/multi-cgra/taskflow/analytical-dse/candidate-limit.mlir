// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl max-cgras-per-task=4 max-candidates=7' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   --mlir-print-op-on-diagnostic=false > %t.diagnostic 2>&1; \
// RUN:   status=$?; FileCheck %s --input-file=%t.diagnostic; check_status=$?; \
// RUN:   test $status -ne 0 -a $check_status -eq 0

module {
  func.func @main(%input: i32) -> i32 {
    %result = taskflow.task @A value_inputs(%input : i32)
        {trip_count = 10 : i64} : (i32) -> i32 {
    ^bb0(%task_input: i32):
      taskflow.yield values(%task_input : i32)
    }
    return %result : i32
  }
}

// CHECK-LABEL: Unknown YAML root key: extensions
// CHECK-NEXT: Unknown YAML root key: simulator
// CHECK-NEXT: {{.*}}candidate-limit.mlir:9:3: error: complete concurrently packable shape space exceeds max-candidates=7; refusing to publish a partial candidate manifest
// CHECK-NEXT:   func.func @main(%input: i32) -> i32 {
// CHECK-NEXT:   ^

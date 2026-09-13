// RUN: mlir-amoeba-opt %s --orchestrate-tasks-on-accelerators --neura-architecture-spec=%S/../../../archspec/architecture_1x2.yaml -o %t.default
// RUN: mlir-amoeba-opt %s --orchestrate-tasks-on-accelerators="orchestration-strategy=routing-critical-path" --neura-architecture-spec=%S/../../../archspec/architecture_1x2.yaml -o %t.explicit
// RUN: diff %t.default %t.explicit
// RUN: FileCheck %s --input-file=%t.explicit

// Each task occupies the entire grid. Their original durations exceed the
// signed 32-bit temporal search horizon, so placement must scale its internal
// time axis while retaining the duration metadata and dependency order.
func.func @large_duration_chain(%seed: i32) -> i32 {
  // CHECK: taskflow.task @producer
  // CHECK-SAME: profile_info = {duration = 2000000000 : i32}
  // CHECK-SAME: cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}, {col = 1 : i32, context_id = 0 : i32, row = 0 : i32}]
  %a = taskflow.task @producer value_inputs(%seed : i32)
      {cgra_count = 2 : i32, profile_info = {duration = 2000000000 : i32}}
      : (i32) -> i32 {
    ^bb0(%x: i32):
      taskflow.yield values(%x : i32)
  }
  // CHECK: taskflow.task @consumer
  // CHECK-SAME: profile_info = {duration = 2000000000 : i32}
  // CHECK-SAME: cgra_positions = [{col = 0 : i32, context_id = 1 : i32, row = 0 : i32}, {col = 1 : i32, context_id = 1 : i32, row = 0 : i32}]
  %b = taskflow.task @consumer value_inputs(%a : i32)
      {cgra_count = 2 : i32, profile_info = {duration = 2000000000 : i32}}
      : (i32) -> i32 {
    ^bb0(%x: i32):
      taskflow.yield values(%x : i32)
  }
  return %b : i32
}

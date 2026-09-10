// XFAIL: *
// RUN: mlir-amoeba-opt %s --resource-aware-task-optimization \
// RUN:   --mlir-print-op-on-diagnostic=false

module {
  func.func @main(%input: i32) -> i32 {
    %result = taskflow.task @A value_inputs(%input : i32)
        {amoeba.analytical_shape_orientation_fixed, cgra_count = 1 : i32,
         cgra_shape = "1x1", trip_count = 1 : i64} : (i32) -> i32 {
    ^bb0(%task_input: i32):
      taskflow.yield values(%task_input : i32)
    }
    return %result : i32
  }
}

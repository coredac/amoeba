// RUN: mlir-amoeba-opt %s --analyze-task-pipeline-interval | FileCheck %s

// T0 -> T1 -> T2. T1 and T2 reuse CGRA (0, 1), so the interval is 3+5.
// CHECK-LABEL: func.func @chain_reuse_cgra1
// CHECK-SAME: task_pipeline_interval_info = {bottleneck_task = "Task_2",
// CHECK-SAME: critical_path = ["Task_1", "Task_2"]
// CHECK-SAME: pipeline_interval = 8 : i32

module {
  func.func @chain_reuse_cgra1(%seed: i32) -> i32 {
    %t0 = taskflow.task @Task_0 value_inputs(%seed : i32)
        {profile_info = {duration = 2 : i32},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t1 = taskflow.task @Task_1 value_inputs(%t0 : i32)
        {profile_info = {duration = 3 : i32},
         task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t2 = taskflow.task @Task_2 value_inputs(%t1 : i32)
        {profile_info = {duration = 5 : i32},
         task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 1 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    return %t2 : i32
  }
}

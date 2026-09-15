// RUN: mlir-amoeba-opt %s --analyze-task-pipeline-interval | FileCheck %s

// T0 and T1 join at T2, then T2 feeds T3. The longest reuse path is T0 -> T2.
// CHECK-LABEL: func.func @diamond_join
// CHECK-SAME: task_pipeline_interval_info = {bottleneck_task = "Task_2",
// CHECK-SAME: critical_path = ["Task_0", "Task_2"]
// CHECK-SAME: pipeline_interval = 10 : i32

module {
  func.func @diamond_join(%seed0: i32, %seed1: i32) -> i32 {
    %t0 = taskflow.task @Task_0 value_inputs(%seed0 : i32)
        {profile_info = {duration = 4 : i32},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t1 = taskflow.task @Task_1 value_inputs(%seed1 : i32)
        {profile_info = {duration = 1 : i32},
         task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t2 = taskflow.task @Task_2 value_inputs(%t0, %t1 : i32, i32)
        {profile_info = {duration = 6 : i32},
         task_orchestration_info = {cgra_positions = [{col = 0 : i32, context_id = 1 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32, i32) -> (i32) {
    ^bb0(%arg0: i32, %arg1: i32):
      taskflow.yield values(%arg0 : i32)
    }

    %t3 = taskflow.task @Task_3 value_inputs(%t2 : i32)
        {profile_info = {duration = 2 : i32},
         task_orchestration_info = {cgra_positions = [{col = 1 : i32, context_id = 1 : i32, row = 0 : i32}],
                                    read_sram_locations = [],
                                    write_sram_locations = []}}
        : (i32) -> (i32) {
    ^bb0(%arg0: i32):
      taskflow.yield values(%arg0 : i32)
    }

    return %t3 : i32
  }
}

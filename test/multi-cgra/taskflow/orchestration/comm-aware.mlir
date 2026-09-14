// Communication-aware placement is opt-in. The large buffer should pull the
// consumer toward its SRAM when --comm-aware is enabled.

// RUN: mlir-amoeba-opt %s '--orchestrate-tasks-on-accelerators=scheduling-mode=spatial' --architecture-spec=%S/../../../archspec/architecture_2x3.yaml -o %t.default.mlir
// RUN: mlir-amoeba-opt %s '--orchestrate-tasks-on-accelerators=scheduling-mode=spatial comm-aware=false' --architecture-spec=%S/../../../archspec/architecture_2x3.yaml -o %t.explicit-default.mlir
// RUN: mlir-amoeba-opt %s '--orchestrate-tasks-on-accelerators=scheduling-mode=spatial comm-aware=true' --architecture-spec=%S/../../../archspec/architecture_2x3.yaml -o %t.comm-aware.mlir
// RUN: diff %t.default.mlir %t.explicit-default.mlir
// RUN: FileCheck %s --input-file=%t.default.mlir --check-prefix=DEFAULT
// RUN: FileCheck %s --input-file=%t.comm-aware.mlir --check-prefix=COMM

module {
  func.func @comm_aware(%small: memref<1xf32>, %big: memref<64xf32>) {
    %small_out = taskflow.task @small_writer
        will_writes(%small : memref<1xf32>)
        [original_write_memrefs(%small : memref<1xf32>)]
        {cgra_count = 1 : i32}
        : (memref<1xf32>) -> (memref<1xf32>) {
      ^bb0(%value: memref<1xf32>):
        taskflow.yield done_writes(%value : memref<1xf32>)
    }

    taskflow.task @filler {cgra_count = 2 : i32, cgra_shape = "2x1"}
        : () -> () {
      taskflow.yield
    }

    %big_out = taskflow.task @big_writer
        will_writes(%big : memref<64xf32>)
        [original_write_memrefs(%big : memref<64xf32>)]
        {cgra_count = 1 : i32}
        : (memref<64xf32>) -> (memref<64xf32>) {
      ^bb0(%value: memref<64xf32>):
        taskflow.yield done_writes(%value : memref<64xf32>)
    }

    taskflow.task @consumer
        will_reads(%small_out, %big_out : memref<1xf32>, memref<64xf32>)
        value_inputs(%small_out, %big_out : memref<1xf32>, memref<64xf32>)
        [original_read_memrefs(%small, %big : memref<1xf32>, memref<64xf32>)]
        {cgra_count = 1 : i32}
        : (memref<1xf32>, memref<64xf32>) -> () {
      ^bb0(%small_arg: memref<1xf32>, %big_arg: memref<64xf32>):
        taskflow.yield
    }

    return
  }
}

// DEFAULT-LABEL: func.func @comm_aware
// DEFAULT: taskflow.task @small_writer
// DEFAULT-SAME: cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 1 : i32}]
// DEFAULT: taskflow.task @filler
// DEFAULT-SAME: cgra_positions = [{col = 2 : i32, context_id = 0 : i32, row = 0 : i32}, {col = 2 : i32, context_id = 0 : i32, row = 1 : i32}]
// DEFAULT: taskflow.task @big_writer
// DEFAULT-SAME: cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 1 : i32}]
// DEFAULT: taskflow.task @consumer
// DEFAULT-SAME: cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}]

// COMM-LABEL: func.func @comm_aware
// COMM: taskflow.task @small_writer
// COMM-SAME: cgra_positions = [{col = 2 : i32, context_id = 0 : i32, row = 1 : i32}]
// COMM: taskflow.task @filler
// COMM-SAME: cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}, {col = 0 : i32, context_id = 0 : i32, row = 1 : i32}]
// COMM: taskflow.task @big_writer
// COMM-SAME: cgra_positions = [{col = 2 : i32, context_id = 0 : i32, row = 0 : i32}]
// COMM: taskflow.task @consumer
// COMM-SAME: cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 1 : i32}]

// RUN: mlir-amoeba-opt %s | mlir-amoeba-opt - -o /dev/null

module {
  func.func @read_write_streams(
      %A: memref<4x4xi32>,
      %C: memref<4x4xi32>
  ) -> memref<4x4xi32> {
    %done_writes = taskflow.task @read_write_streams
        will_reads(%A : memref<4x4xi32>)
        will_writes(%C : memref<4x4xi32>)
        [original_read_memrefs(%A : memref<4x4xi32>),
         original_write_memrefs(%C : memref<4x4xi32>)]
        : (memref<4x4xi32>, memref<4x4xi32>)
       -> (memref<4x4xi32>) {
    ^bb0(
        %task_A: memref<4x4xi32>,
        %task_C: memref<4x4xi32>
    ):
      %a:4 = taskflow.stream_read %task_A maps [
        affine_map<(i) -> (i, 0)>,
        affine_map<(i) -> (i, 1)>,
        affine_map<(i) -> (i, 2)>,
        affine_map<(i) -> (i, 3)>
      ] : memref<4x4xi32> -> (i32, i32, i32, i32)

      taskflow.stream_write(
        %a#0, %a#1, %a#2, %a#3 : i32, i32, i32, i32
      ) to %task_C maps [
        affine_map<(i) -> (i, 0)>,
        affine_map<(i) -> (i, 1)>,
        affine_map<(i) -> (i, 2)>,
        affine_map<(i) -> (i, 3)>
      ] : memref<4x4xi32>

      taskflow.yield done_writes(%task_C : memref<4x4xi32>)
    }

    return %done_writes : memref<4x4xi32>
  }
}
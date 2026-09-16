// Shared analytical task metadata preserves nested static counter trip counts
// and rejects dynamic bounds before publishing a candidate manifest.
//
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=function=static output=%t.static.jsonl max-cgras-per-task=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.static.jsonl --check-prefix=STATIC
// RUN: ! mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=function=dynamic output=%t.dynamic.jsonl max-cgras-per-task=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null > %t.dynamic.err 2>&1
// RUN: FileCheck %s --input-file=%t.dynamic.err --check-prefix=DYNAMIC
// RUN: ! test -e %t.dynamic.jsonl

module {
  func.func @static(%a: memref<16xf32>) {
    %a_read, %a_write = taskflow.task @Static
        will_reads(%a : memref<16xf32>)
        will_writes(%a : memref<16xf32>)
        [original_read_memrefs(%a : memref<16xf32>),
         original_write_memrefs(%a : memref<16xf32>)]
        : (memref<16xf32>, memref<16xf32>)
       -> (memref<16xf32>, memref<16xf32>) {
    ^bb0(%input: memref<16xf32>, %output: memref<16xf32>):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %outer = taskflow.counter from %c0 to %c2 step %c1 : index
      %inner = taskflow.counter parent(%outer : index)
          from %c0 to %c3 step %c1 : index
      taskflow.yield done_reads(%input : memref<16xf32>)
                     done_writes(%output : memref<16xf32>)
    }
    return
  }

  func.func @dynamic(%a: memref<16xf32>, %upper: index) {
    %a_read, %a_write = taskflow.task @Dynamic
        will_reads(%a : memref<16xf32>)
        will_writes(%a : memref<16xf32>)
        value_inputs(%upper : index)
        [original_read_memrefs(%a : memref<16xf32>),
         original_write_memrefs(%a : memref<16xf32>)]
        : (memref<16xf32>, memref<16xf32>, index)
       -> (memref<16xf32>, memref<16xf32>) {
    ^bb0(%input: memref<16xf32>, %output: memref<16xf32>, %limit: index):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %limit step %c1 : index
      taskflow.yield done_reads(%input : memref<16xf32>)
                     done_writes(%output : memref<16xf32>)
    }
    return
  }
}

// STATIC: {"candidate_id":"candidate-0","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"Static","trip_count":6}]}
// STATIC-NEXT: {"candidate_count":1,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}
// DYNAMIC: error: task Dynamic requires constant counter bounds, a positive step, a non-empty range, and a trip count within int64; add an explicit positive trip_count or resolve the counter bounds first

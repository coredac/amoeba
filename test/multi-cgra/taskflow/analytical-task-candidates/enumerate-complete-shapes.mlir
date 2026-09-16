// The default per-task footprint spans the physical 4x4 grid, and every
// fixed-rotation rectangular shape is part of the candidate space.
//
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o %t.bound.mlir
// RUN: FileCheck %s --input-file=%t.candidates.jsonl --check-prefix=CANDIDATES
// RUN: FileCheck %s --input-file=%t.bound.mlir --check-prefix=BOUND

module {
  func.func @main(%a: memref<16xf32>) {
    %a_read, %a_write = taskflow.task @A
        will_reads(%a : memref<16xf32>)
        will_writes(%a : memref<16xf32>)
        [original_read_memrefs(%a : memref<16xf32>),
         original_write_memrefs(%a : memref<16xf32>)]
        : (memref<16xf32>, memref<16xf32>)
       -> (memref<16xf32>, memref<16xf32>) {
    ^bb0(%input: memref<16xf32>, %output: memref<16xf32>):
      %c0 = arith.constant 0 : index
      %c10 = arith.constant 10 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %c10 step %c1 : index
      taskflow.yield done_reads(%input : memref<16xf32>)
                     done_writes(%output : memref<16xf32>)
    }
    return
  }
}

// CANDIDATES-LABEL: {"architecture":
// CANDIDATES-SAME: "max_cgras_per_task":16
// CANDIDATES-SAME: "shape_pruning_policy":"none"
// CANDIDATES: {"candidate_id":"candidate-15","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":16,"cgra_shape":"4x4","cols":4,"kind":"rect","mapper_tile_cols":16,"mapper_tile_rows":16,"rows":4},"task":"A","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_count":16,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}
// BOUND: amoeba.source_task_body_sha256 = "f4e821880514d53c4c0fdd5866807dca5eeb44b23220207881ad7846ef3308fe"

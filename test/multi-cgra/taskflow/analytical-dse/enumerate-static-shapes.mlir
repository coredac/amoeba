// On a 4x4 grid with at most four CGRAs per task, each task has eight oriented
// rectangular shapes. Of the 64 ordered pairs, exactly two cannot coexist:
// (A=1x4, B=4x1) and (A=4x1, B=1x4). Their total area fits, so excluding them
// verifies exact fixed-orientation packing.

// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl max-cgras-per-task=4 max-candidates=62' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o %t.bound.mlir
// RUN: FileCheck %s --input-file=%t.candidates.jsonl --check-prefix=CANDIDATES
// RUN: FileCheck %s --input-file=%t.bound.mlir --check-prefix=BOUND

// CANDIDATES: "spec_sha256":"{{[0-9a-f]+}}"
// CANDIDATES: "spatial_capacity_policy":"all-tasks-simultaneous-exact-pack"
// CANDIDATES: "body_sha256":"{{[0-9a-f]+}}"
// CANDIDATES: "trip_count":10
// CANDIDATES: "candidate_id":"candidate-0"
// CANDIDATES-NOT: "cgra_shape":"1x4"{{.*}}"cgra_shape":"4x1"
// CANDIDATES-NOT: "cgra_shape":"4x1"{{.*}}"cgra_shape":"1x4"
// CANDIDATES: "candidate_count":62
// BOUND-COUNT-2: amoeba.source_task_body_sha256 = "{{[0-9a-f]+}}"

module {
  func.func @main(%a: memref<16xf32>, %b: memref<16xf32>) {
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
    %b_read, %b_write = taskflow.task @B
        will_reads(%b : memref<16xf32>)
        will_writes(%b : memref<16xf32>)
        [original_read_memrefs(%b : memref<16xf32>),
         original_write_memrefs(%b : memref<16xf32>)]
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

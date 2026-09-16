// Candidate limits fail before a partial manifest can be published, while an
// explicit per-task limit remains part of the spatial search contract.
//
// RUN: ! mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.limit.jsonl max-candidates=15' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null > %t.limit.err 2>&1
// RUN: FileCheck %s --input-file=%t.limit.err --check-prefix=LIMIT
// RUN: ! test -e %t.limit.jsonl
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.one.jsonl max-cgras-per-task=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.one.jsonl --check-prefix=ONE

module {
  func.func @main(%a: memref<16xf32>) {
    %a_read, %a_write = taskflow.task @A
        will_reads(%a : memref<16xf32>)
        will_writes(%a : memref<16xf32>)
        [original_read_memrefs(%a : memref<16xf32>),
         original_write_memrefs(%a : memref<16xf32>)]
        {trip_count = 10 : i64}
        : (memref<16xf32>, memref<16xf32>)
       -> (memref<16xf32>, memref<16xf32>) {
    ^bb0(%input: memref<16xf32>, %output: memref<16xf32>):
      %c0 = "neura.constant"() <{value = 0 : index}> : () -> index
      %c1 = "neura.constant"() <{value = 1 : index}> : () -> index
      %c2 = "neura.constant"() <{value = 2 : index}> : () -> index
      %c3 = "neura.constant"() <{value = 3 : index}> : () -> index
      %c4 = "neura.constant"() <{value = 4 : index}> : () -> index
      %c5 = "neura.constant"() <{value = 5 : index}> : () -> index
      %c6 = "neura.constant"() <{value = 6 : index}> : () -> index
      %c7 = "neura.constant"() <{value = 7 : index}> : () -> index
      taskflow.yield done_reads(%input : memref<16xf32>)
                     done_writes(%output : memref<16xf32>)
    }
    return
  }
}

// LIMIT: complete concurrently packable shape space exceeds max-candidates=15
// ONE-LABEL: {"architecture":{"grid_cols":4,"grid_rows":4,"per_cgra_tile_cols":4,"per_cgra_tile_rows":4,"spec_sha256":"1b96d6d3741805b057add7db66296c9c73cc45ea4ef0263e0eea5adb4f4360d6"}
// ONE-SAME: "max_cgras_per_task":1
// ONE-SAME: "shape_pruning_policy":"none"
// ONE: {"candidate_id":"candidate-0","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10}]}
// ONE-NEXT: {"candidate_count":1,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}

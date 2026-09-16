// Two tasks share a 1x2 physical grid. The only simultaneously packable
// tuple gives each task one CGRA; assigning the full grid to either task is
// rejected by exact concurrent packing.
//
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_1x2.yaml \
// RUN:   -o %t.bound.mlir
// RUN: FileCheck %s --input-file=%t.candidates.jsonl --check-prefix=CANDIDATES
// RUN: FileCheck %s --input-file=%t.bound.mlir --check-prefix=BOUND
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.geometry.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null
// RUN: FileCheck %s --input-file=%t.geometry.jsonl --check-prefix=GEOMETRY

module {
  func.func @main(%a: memref<16xf32>, %b: memref<16xf32>) {
    %a_read, %a_write = taskflow.task @A
        will_reads(%a : memref<16xf32>)
        will_writes(%a : memref<16xf32>)
        [original_read_memrefs(%a : memref<16xf32>),
         original_write_memrefs(%a : memref<16xf32>)]
        {trip_count = 10 : i64}
        : (memref<16xf32>, memref<16xf32>)
       -> (memref<16xf32>, memref<16xf32>) {
    ^bb0(%input: memref<16xf32>, %output: memref<16xf32>):
      taskflow.yield done_reads(%input : memref<16xf32>)
                     done_writes(%output : memref<16xf32>)
    }
    %b_read, %b_write = taskflow.task @B
        will_reads(%b : memref<16xf32>)
        will_writes(%b : memref<16xf32>)
        [original_read_memrefs(%b : memref<16xf32>),
         original_write_memrefs(%b : memref<16xf32>)]
        {trip_count = 10 : i64}
        : (memref<16xf32>, memref<16xf32>)
       -> (memref<16xf32>, memref<16xf32>) {
    ^bb0(%input: memref<16xf32>, %output: memref<16xf32>):
      taskflow.yield done_reads(%input : memref<16xf32>)
                     done_writes(%output : memref<16xf32>)
    }
    return
  }
}

// CANDIDATES-LABEL: {"architecture":{"grid_cols":2,"grid_rows":1,"per_cgra_tile_cols":4,"per_cgra_tile_rows":4,"spec_sha256":"e7dc2d3a55b924f603c741f9f28d9c1616642b831b2bed10b2c7ba32eafb01ba"}
// CANDIDATES-SAME: "cost_queries":[{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"A"},{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"B"}]
// CANDIDATES-SAME: "max_cgras_per_task":2
// CANDIDATES-SAME: "shape_pruning_policy":"none"
// CANDIDATES: {"candidate_id":"candidate-0","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10},{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"B","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_count":1,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}
// BOUND: amoeba.source_task_body_sha256 = "8fcfb8c42698245d9664f7fc44a11354433beede74c4eac7c70123bffaae0047"
// A 1x4 row and a 4x1 column consume only half of a 4x4 grid by area, but
// fixed rotation makes them geometrically impossible to place together.
// GEOMETRY-LABEL: {"architecture":
// GEOMETRY-NOT: "task_shapes":[{"shape":{"cgra_count":4,"cgra_shape":"1x4","cols":4,"kind":"rect","mapper_tile_cols":16,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10},{"shape":{"cgra_count":4,"cgra_shape":"4x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":16,"rows":4},"task":"B","trip_count":10}]
// GEOMETRY: {"candidate_count":

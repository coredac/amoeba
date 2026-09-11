// The default per-task limit is resolved from the 4x4 physical grid as 16.
// Operation-count pruning still caps these counter-only tasks at one CGRA, so
// the complete feasible tuple space contains one candidate and two 4x4 mapper
// queries rather than inventing useful work for the larger rectangles.

// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o %t.bound.mlir
// RUN: FileCheck %s --input-file=%t.candidates.jsonl --check-prefix=CANDIDATES
// RUN: FileCheck %s --input-file=%t.bound.mlir --check-prefix=BOUND

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

// CANDIDATES-LABEL: {"architecture":{"grid_cols":4,"grid_rows":4,"per_cgra_tile_cols":4,"per_cgra_tile_rows":4,"spec_sha256":"1b96d6d3741805b057add7db66296c9c73cc45ea4ef0263e0eea5adb4f4360d6"},"cost_queries":[{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"A"},{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"B"}],"fixed_axes":{"communication":"not-scored","fission":"factor-1","fusion":"identity","placement":"exact-fit-required-coordinates-downstream-heuristic","temporal_order":"downstream-heuristic","tiling":"factor-1"},"function":"main","max_cgras_per_task":16,"record_type":"header","schema":"amoeba-analytical-task-candidates","search_scope":"static-shape-concurrent-fit","shape_policy":"static-oriented-rectangles","shape_pruning_policy":"half-full-two-cgra-floor-then-ceil-ops-per-cgra","spatial_capacity_policy":"all-tasks-simultaneous-exact-pack","tasks":[{"body_sha256":"f4e821880514d53c4c0fdd5866807dca5eeb44b23220207881ad7846ef3308fe","materialized_operation_count":0,"maximum_physical_cgras":1,"task":"A","trip_count":10},{"body_sha256":"f4e821880514d53c4c0fdd5866807dca5eeb44b23220207881ad7846ef3308fe","materialized_operation_count":0,"maximum_physical_cgras":1,"task":"B","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_id":"candidate-0","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10},{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"B","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_count":1,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}
// BOUND-COUNT-2: amoeba.source_task_body_sha256 = "f4e821880514d53c4c0fdd5866807dca5eeb44b23220207881ad7846ef3308fe"

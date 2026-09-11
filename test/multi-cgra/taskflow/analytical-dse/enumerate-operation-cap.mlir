// A half-full four-by-four CGRA retains both oriented two-CGRA shapes.
// The empty second task uses one CGRA. The default global footprint reaches
// the whole 4x4 physical grid; operation-count pruning makes three candidates.
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o %t.bound.mlir
// RUN: FileCheck %s --input-file=%t.candidates.jsonl --check-prefix=CANDIDATES
// RUN: FileCheck %s --input-file=%t.bound.mlir --check-prefix=BOUND
// RUN: ! mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.limit.jsonl max-candidates=2' \
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

// CANDIDATES-LABEL: {"architecture":{"grid_cols":4,"grid_rows":4,"per_cgra_tile_cols":4,"per_cgra_tile_rows":4,"spec_sha256":"1b96d6d3741805b057add7db66296c9c73cc45ea4ef0263e0eea5adb4f4360d6"},"cost_queries":[{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"A"},{"mapper_tile_cols":8,"mapper_tile_rows":4,"task":"A"},{"mapper_tile_cols":4,"mapper_tile_rows":8,"task":"A"},{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"B"}],"fixed_axes":{"communication":"not-scored","fission":"factor-1","fusion":"identity","placement":"exact-fit-required-coordinates-downstream-heuristic","temporal_order":"downstream-heuristic","tiling":"factor-1"},"function":"main","max_cgras_per_task":16,"record_type":"header","schema":"amoeba-analytical-task-candidates","search_scope":"static-shape-concurrent-fit","shape_policy":"static-oriented-rectangles","shape_pruning_policy":"half-full-two-cgra-floor-then-ceil-ops-per-cgra","spatial_capacity_policy":"all-tasks-simultaneous-exact-pack","tasks":[{"body_sha256":"0eee37200729f384c599f95c11f603cc21b3043ce0fca7817efcb65510edb05d","materialized_operation_count":8,"maximum_physical_cgras":2,"task":"A","trip_count":10},{"body_sha256":"8fcfb8c42698245d9664f7fc44a11354433beede74c4eac7c70123bffaae0047","materialized_operation_count":0,"maximum_physical_cgras":1,"task":"B","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_id":"candidate-0","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10},{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"B","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_id":"candidate-1","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":2,"cgra_shape":"1x2","cols":2,"kind":"rect","mapper_tile_cols":8,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10},{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"B","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_id":"candidate-2","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":2,"cgra_shape":"2x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":8,"rows":2},"task":"A","trip_count":10},{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"B","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_count":3,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}
// BOUND: amoeba.source_task_body_sha256 = "0eee37200729f384c599f95c11f603cc21b3043ce0fca7817efcb65510edb05d"
// BOUND: amoeba.source_task_body_sha256 = "8fcfb8c42698245d9664f7fc44a11354433beede74c4eac7c70123bffaae0047"
// LIMIT: complete concurrently packable shape space exceeds max-candidates=2
// ONE-LABEL: {"architecture":{"grid_cols":4,"grid_rows":4,"per_cgra_tile_cols":4,"per_cgra_tile_rows":4,"spec_sha256":"1b96d6d3741805b057add7db66296c9c73cc45ea4ef0263e0eea5adb4f4360d6"},"cost_queries":[{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"A"},{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"B"}],"fixed_axes":{"communication":"not-scored","fission":"factor-1","fusion":"identity","placement":"exact-fit-required-coordinates-downstream-heuristic","temporal_order":"downstream-heuristic","tiling":"factor-1"},"function":"main","max_cgras_per_task":1,"record_type":"header","schema":"amoeba-analytical-task-candidates","search_scope":"static-shape-concurrent-fit","shape_policy":"static-oriented-rectangles","shape_pruning_policy":"half-full-two-cgra-floor-then-ceil-ops-per-cgra","spatial_capacity_policy":"all-tasks-simultaneous-exact-pack","tasks":[{"body_sha256":"0eee37200729f384c599f95c11f603cc21b3043ce0fca7817efcb65510edb05d","materialized_operation_count":8,"maximum_physical_cgras":1,"task":"A","trip_count":10},{"body_sha256":"8fcfb8c42698245d9664f7fc44a11354433beede74c4eac7c70123bffaae0047","materialized_operation_count":0,"maximum_physical_cgras":1,"task":"B","trip_count":10}]}
// ONE-NEXT: {"candidate_id":"candidate-0","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10},{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"B","trip_count":10}]}
// ONE-NEXT: {"candidate_count":1,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}

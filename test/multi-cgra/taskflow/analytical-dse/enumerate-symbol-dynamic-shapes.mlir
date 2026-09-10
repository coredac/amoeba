// RUN: mlir-amoeba-opt %s \
// RUN:   --classify-task-and-counter \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_1x2.yaml \
// RUN:   --mlir-print-op-on-diagnostic=false > %t.program 2>&1
// RUN: FileCheck %s --input-file=%t.program --check-prefix=PROGRAM
// RUN: FileCheck %s --input-file=%t.candidates.jsonl --check-prefix=CANDIDATES
// RUN: mlir-amoeba-opt %s \
// RUN:   --classify-task-and-counter \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t.candidates.jsonl candidate-id=candidate-1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_1x2.yaml \
// RUN:   --mlir-print-op-on-diagnostic=false > %t.materialized 2>&1
// RUN: FileCheck %s --input-file=%t.materialized --check-prefix=MATERIALIZED

module {
  func.func @main(%upper: index, %input: i32) -> i32 {
    %result = taskflow.task @A value_inputs(%upper, %input : index, i32)
        : (index, i32) -> i32 {
    ^bb0(%task_upper: index, %task_input: i32):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %task_upper step %c1 : index
      taskflow.yield values(%task_input : i32)
    }
    return %result : i32
  }
}

// PROGRAM-LABEL: Unknown YAML root key: extensions
// PROGRAM-NEXT: Unknown YAML root key: simulator
// PROGRAM-NEXT: Warning: Overwriting existing register file cluster (0) in Tile 0
// PROGRAM-NEXT: Warning: Overwriting existing register file cluster (4) in Tile 4
// PROGRAM-NEXT: Warning: Overwriting existing register file cluster (8) in Tile 8
// PROGRAM-NEXT: Warning: Overwriting existing register file cluster (12) in Tile 12
// PROGRAM-NEXT: Warning: Overwriting existing register file cluster (1) in Tile 1
// PROGRAM-NEXT: Warning: Overwriting existing register file cluster (2) in Tile 2
// PROGRAM-NEXT: Warning: Overwriting existing register file cluster (3) in Tile 3
// PROGRAM-NEXT: [AnalyticalTaskDSE] enumerated all 2 concurrently packable shape candidates into {{.*}}.candidates.jsonl
// PROGRAM-NEXT: module {
// PROGRAM-NEXT:   func.func @main(%arg0: index, %arg1: i32) -> i32 {
// PROGRAM-NEXT:     %value_outputs = taskflow.task @A value_inputs(%arg0, %arg1 : index, i32) {amoeba.source_task_body_sha256 = "81f07c2a6bc70d613116e8c1e75c08234e90c4989eb36d13c238d8120ef8bc11"} : (index, i32) -> (i32) {
// PROGRAM-NEXT:     ^bb0(%arg2: index, %arg3: i32):
// PROGRAM-NEXT:       %c0 = arith.constant 0 : index
// PROGRAM-NEXT:       %c1 = arith.constant 1 : index
// PROGRAM-NEXT:       %0 = taskflow.counter from %c0 to %arg2 step %c1 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// PROGRAM-NEXT:       taskflow.yield values(%arg3 : i32)
// PROGRAM-NEXT:     }
// PROGRAM-NEXT:     return %value_outputs : i32
// PROGRAM-NEXT:   }
// PROGRAM-NEXT: }
// PROGRAM-EMPTY

// CANDIDATES-LABEL: {"architecture":{"grid_cols":2,"grid_rows":1,"per_cgra_tile_cols":4,"per_cgra_tile_rows":4,"spec_sha256":"e7dc2d3a55b924f603c741f9f28d9c1616642b831b2bed10b2c7ba32eafb01ba"},"cost_queries":[{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"A"},{"mapper_tile_cols":8,"mapper_tile_rows":4,"task":"A"}],"fixed_axes":{"communication":"not-scored","fission":"factor-1","fusion":"identity","placement":"exact-fit-required-coordinates-downstream-heuristic","temporal_order":"downstream-heuristic","tiling":"factor-1"},"function":"main","record_type":"header","schema":"amoeba-analytical-task-candidates","search_scope":"static-shape-concurrent-fit","shape_policy":"static-oriented-rectangles","spatial_capacity_policy":"all-tasks-simultaneous-exact-pack","tasks":[{"body_sha256":"81f07c2a6bc70d613116e8c1e75c08234e90c4989eb36d13c238d8120ef8bc11","task":"A","trip_count_kind":"symbol_dynamic"}]}
// CANDIDATES-NEXT: {"candidate_id":"candidate-0","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count_kind":"symbol_dynamic"}]}
// CANDIDATES-NEXT: {"candidate_id":"candidate-1","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":2,"cgra_shape":"1x2","cols":2,"kind":"rect","mapper_tile_cols":8,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count_kind":"symbol_dynamic"}]}
// CANDIDATES-NEXT: {"candidate_count":2,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}
// MATERIALIZED-LABEL: module {
// MATERIALIZED-NEXT:   func.func @main(%arg0: index, %arg1: i32) -> i32 attributes {analytical_task_candidate_id = "candidate-1", analytical_task_candidate_scope = "static-shape-concurrent-fit"} {
// MATERIALIZED-NEXT:     %value_outputs = taskflow.task @A value_inputs(%arg0, %arg1 : index, i32) {amoeba.analytical_shape_orientation_fixed, cgra_count = 2 : i32, cgra_shape = "1x2"} : (index, i32) -> (i32) {
// MATERIALIZED-NEXT:     ^bb0(%arg2: index, %arg3: i32):
// MATERIALIZED-NEXT:       %c0 = arith.constant 0 : index
// MATERIALIZED-NEXT:       %c1 = arith.constant 1 : index
// MATERIALIZED-NEXT:       %0 = taskflow.counter from %c0 to %arg2 step %c1 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 0 : i32} : index
// MATERIALIZED-NEXT:       taskflow.yield values(%arg3 : i32)
// MATERIALIZED-NEXT:     }
// MATERIALIZED-NEXT:     return %value_outputs : i32
// MATERIALIZED-NEXT:   }
// MATERIALIZED-NEXT: }
// MATERIALIZED-EMPTY

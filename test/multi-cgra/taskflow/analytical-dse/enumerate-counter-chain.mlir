// RUN: mlir-amoeba-opt %s \
// RUN:   --classify-task-and-counter \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_1x2.yaml \
// RUN:   --mlir-print-op-on-diagnostic=false > %t.program 2>&1
// RUN: FileCheck %s --input-file=%t.program --check-prefix=PROGRAM
// RUN: FileCheck %s --input-file=%t.candidates.jsonl --check-prefix=CANDIDATES

module {
  func.func @main() {
    taskflow.task @A : () -> () {
    ^bb0:
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c3 = arith.constant 3 : index
      %c5 = arith.constant 5 : index
      %root = taskflow.counter from %c0 to %c2 step %c1 : index
      %child3 = taskflow.counter parent(%root : index) from %c0 to %c3 step %c1 : index
      %child5 = taskflow.counter parent(%root : index) from %c0 to %c5 step %c1 : index
      taskflow.yield
    }
    return
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
// PROGRAM-NEXT:   func.func @main() {
// PROGRAM-NEXT:     taskflow.task @A {amoeba.source_task_body_sha256 = "018eb7855f916a014cd19595f4b66fbc3f182c74d51eb59f1b11d02aab9b7d6b"} : () -> () {
// PROGRAM-NEXT:       %c0 = arith.constant 0 : index
// PROGRAM-NEXT:       %c1 = arith.constant 1 : index
// PROGRAM-NEXT:       %c2 = arith.constant 2 : index
// PROGRAM-NEXT:       %c3 = arith.constant 3 : index
// PROGRAM-NEXT:       %c5 = arith.constant 5 : index
// PROGRAM-NEXT:       %0 = taskflow.counter from %c0 to %c2 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
// PROGRAM-NEXT:       %1 = taskflow.counter parent(%0 : index) from %c0 to %c3 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 1 : i32} : index
// PROGRAM-NEXT:       %2 = taskflow.counter parent(%0 : index) from %c0 to %c5 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
// PROGRAM-NEXT:       taskflow.yield
// PROGRAM-NEXT:     }
// PROGRAM-NEXT:     return
// PROGRAM-NEXT:   }
// PROGRAM-NEXT: }
// PROGRAM-EMPTY

// CANDIDATES-LABEL: {"architecture":{"grid_cols":2,"grid_rows":1,"per_cgra_tile_cols":4,"per_cgra_tile_rows":4,"spec_sha256":"e7dc2d3a55b924f603c741f9f28d9c1616642b831b2bed10b2c7ba32eafb01ba"},"cost_queries":[{"mapper_tile_cols":4,"mapper_tile_rows":4,"task":"A"},{"mapper_tile_cols":8,"mapper_tile_rows":4,"task":"A"}],"fixed_axes":{"communication":"not-scored","fission":"factor-1","fusion":"identity","placement":"exact-fit-required-coordinates-downstream-heuristic","temporal_order":"downstream-heuristic","tiling":"factor-1"},"function":"main","record_type":"header","schema":"amoeba-analytical-task-candidates","search_scope":"static-shape-concurrent-fit","shape_policy":"static-oriented-rectangles","spatial_capacity_policy":"all-tasks-simultaneous-exact-pack","tasks":[{"body_sha256":"018eb7855f916a014cd19595f4b66fbc3f182c74d51eb59f1b11d02aab9b7d6b","task":"A","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_id":"candidate-0","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":1,"cgra_shape":"1x1","cols":1,"kind":"rect","mapper_tile_cols":4,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_id":"candidate-1","record_type":"candidate","schema":"amoeba-analytical-task-candidates","task_shapes":[{"shape":{"cgra_count":2,"cgra_shape":"1x2","cols":2,"kind":"rect","mapper_tile_cols":8,"mapper_tile_rows":4,"rows":1},"task":"A","trip_count":10}]}
// CANDIDATES-NEXT: {"candidate_count":2,"record_type":"footer","schema":"amoeba-analytical-task-candidates"}

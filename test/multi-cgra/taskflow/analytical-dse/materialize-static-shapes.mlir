// Candidate 6 selects the vertical two-CGRA shape for A and one CGRA for B.
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl max-cgras-per-task=2' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null
// RUN: mlir-amoeba-opt %s \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t.candidates.jsonl candidate-id=candidate-6' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o %t.materialized.mlir
// RUN: FileCheck %s --input-file=%t.materialized.mlir --check-prefix=MATERIALIZED
// RUN: ! mlir-amoeba-opt %t.materialized.mlir \
// RUN:   --resource-aware-task-optimization \
// RUN:   -o /dev/null > %t.no-reallocate.err 2>&1
// RUN: FileCheck %s --input-file=%t.no-reallocate.err --check-prefix=NO-REALLOCATE
// RUN: mlir-amoeba-opt %t.materialized.mlir \
// RUN:   '--orchestrate-tasks-on-accelerators=orchestration-strategy=analytical-dse-spatial scheduling-mode=spatial' \
// RUN:   --neura-architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o %t.orchestrated.mlir
// RUN: FileCheck %s --input-file=%t.orchestrated.mlir --check-prefix=ORCHESTRATED
// RUN: sed '/"record_type":"candidate"/d' %t.candidates.jsonl > %t.incomplete.jsonl
// RUN: ! mlir-amoeba-opt %s \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t.incomplete.jsonl candidate-id=candidate-0' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null > %t.incomplete.err 2>&1
// RUN: FileCheck %s --input-file=%t.incomplete.err --check-prefix=INCOMPLETE
// RUN: ! mlir-amoeba-opt %s \
// RUN:   '--orchestrate-tasks-on-accelerators=orchestration-strategy=analytical-dse-spatial scheduling-mode=spatial' \
// RUN:   --neura-architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null > %t.unbound.err 2>&1
// RUN: FileCheck %s --input-file=%t.unbound.err --check-prefix=UNBOUND
// RUN: sed 's/cgra_shape = "2x1"/cgra_shape = "2x1[(0,0)(0,1)]"/' \
// RUN:   %t.materialized.mlir > %t.nonrect.mlir
// RUN: ! mlir-amoeba-opt %t.nonrect.mlir \
// RUN:   '--orchestrate-tasks-on-accelerators=orchestration-strategy=analytical-dse-spatial scheduling-mode=spatial' \
// RUN:   --neura-architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null > %t.nonrect.err 2>&1
// RUN: FileCheck %s --input-file=%t.nonrect.err --check-prefix=NONRECT

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

// MATERIALIZED-LABEL: func.func @main
// MATERIALIZED-SAME: analytical_task_candidate_id = "candidate-6"
// MATERIALIZED: taskflow.task @A
// MATERIALIZED-SAME: amoeba.analytical_shape_orientation_fixed
// MATERIALIZED-SAME: cgra_count = 2 : i32
// MATERIALIZED-SAME: cgra_shape = "2x1"
// MATERIALIZED: taskflow.task @B
// MATERIALIZED-SAME: amoeba.analytical_shape_orientation_fixed
// MATERIALIZED-SAME: cgra_count = 1 : i32
// MATERIALIZED-SAME: cgra_shape = "1x1"
// ORCHESTRATED-LABEL: func.func @main
// ORCHESTRATED-SAME: analytical_task_candidate_id = "candidate-6"
// ORCHESTRATED: taskflow.task @A
// ORCHESTRATED-NOT: amoeba.analytical_shape_orientation_fixed
// ORCHESTRATED-SAME: cgra_positions = [{col = 0 : i32, context_id = 0 : i32, row = 0 : i32}, {col = 0 : i32, context_id = 0 : i32, row = 1 : i32}]
// ORCHESTRATED: taskflow.task @B
// ORCHESTRATED-NOT: amoeba.analytical_shape_orientation_fixed
// ORCHESTRATED-SAME: cgra_positions = [{col = 1 : i32, context_id = 0 : i32, row = 0 : i32}]
// NO-REALLOCATE: resource-aware-task-optimization cannot run after an analytical task candidate has fixed resource shapes
// INCOMPLETE: candidate manifest count does not match the complete concurrently packable shape space
// UNBOUND: analytical DSE spatial orchestration requires a non-empty analytical_task_candidate_id attribute
// NONRECT: invalid fixed cgra_shape: cgra_shape has invalid rectangle dimensions

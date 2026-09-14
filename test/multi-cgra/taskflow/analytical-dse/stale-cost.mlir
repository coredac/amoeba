// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o %t.bound.mlir
// RUN: candidate_sha=$(sha256sum %t.candidates.jsonl | cut -d' ' -f1) && \
// RUN: architecture_sha=$(sha256sum %S/../../../archspec/architecture_4x4.yaml | cut -d' ' -f1) && \
// RUN: sed -e "s/CANDIDATE_SHA_PLACEHOLDER/$candidate_sha/g" \
// RUN:     -e "s/ARCHITECTURE_SHA_PLACEHOLDER/$architecture_sha/g" \
// RUN:     %S/stale-cost.json > %t.stale-cost.json
// RUN: ! mlir-amoeba-opt %s \
// RUN:   '--score-analytical-task-candidates=candidates=%t.candidates.jsonl cost-file=%t.stale-cost.json output=%t.scores.jsonl top-k=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null > %t.stale.err 2>&1
// RUN: FileCheck %s --input-file=%t.stale.err --check-prefix=STALE
// RUN: ! test -e %t.scores.jsonl

// A catalogue with stale task provenance must be rejected before any score is
// published, even when its manifest and architecture hashes are current.
// STALE: cost catalogue task provenance does not bind the current task body to its source DFG

module {
  func.func @main(%a: memref<16xf32>) {
    %read, %write = taskflow.task @A
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
    return
  }
}

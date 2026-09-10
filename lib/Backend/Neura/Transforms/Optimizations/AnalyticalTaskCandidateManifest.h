//===- AnalyticalTaskCandidateManifest.h ----------------------*- C++ -*-===//
//
// Declares strict parsing and validation of static task-shape manifests.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_TASK_CANDIDATE_MANIFEST_H
#define AMOEBA_ANALYTICAL_TASK_CANDIDATE_MANIFEST_H

#include "AnalyticalTaskCandidateSpace.h"

#include "NeuraDialect/Architecture/Architecture.h"

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

// Stores the values that define a finite candidate space.
struct ManifestHeader {
  std::string function;
  std::string architectureSha256;
  int64_t gridRows = 0;
  int64_t gridCols = 0;
  int64_t perCgraRows = 0;
  int64_t perCgraCols = 0;
};

// Stores the record count that closes a manifest.
struct ManifestFooter {
  uint64_t candidateCount = 0;
};

using CandidateConsumer =
    llvm::function_ref<bool(uint64_t, const Candidate &, std::string &)>;

// For example, this candidate JSONL record:
//
//   {"record_type":"candidate","schema":"amoeba-analytical-task-candidates",
//    "candidate_id":"candidate-3","task_shapes":[{"task":"A",
//    "trip_count":4,"shape":{"kind":"rect","rows":1,"cols":2,
//    "cgra_count":2,"cgra_shape":"1x2","mapper_tile_rows":4,
//    "mapper_tile_cols":8}}]}
//
// is parsed into a Candidate with these values:
//
//   candidate.id == "candidate-3"
//   candidate.choices[0].task == "A"
//   candidate.choices[0].tripCount == 4
//   candidate.choices[0].shape == RectShape{1, 2, 4, 8}
//
// A successful parse guarantees header/candidate/footer ordering, current IR
// and architecture identity, canonical candidate order, exact concurrent
// packing, and completeness of the finite candidate space.
bool readCandidateManifest(llvm::StringRef path, llvm::ArrayRef<TaskFact> tasks,
                           llvm::StringRef expectedFunction,
                           const ::mlir::neura::Architecture &architecture,
                           CandidateConsumer consume, ManifestHeader &header,
                           ManifestFooter &footer, std::string &error);

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_TASK_CANDIDATE_MANIFEST_H

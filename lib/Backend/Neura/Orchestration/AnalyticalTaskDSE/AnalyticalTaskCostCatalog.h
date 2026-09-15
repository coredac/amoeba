//===- AnalyticalTaskCostCatalog.h ---------------------------*- C++ -*-===//
//
// Declares the validated task-shape ML cost oracle used by analytical DSE.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_TASK_COST_CATALOG_H
#define AMOEBA_ANALYTICAL_TASK_COST_CATALOG_H

#include "SpatialTaskCandidateSpace.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <tuple>

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

inline constexpr llvm::StringLiteral kCostSchema = "amoeba-task-shape-cost";
inline constexpr llvm::StringLiteral kScoreSchema =
    "amoeba-analytical-task-scores";
inline constexpr llvm::StringLiteral kScoreModel =
    "spatial-temporal-scheduler-makespan";

// Stores one predictor result. Unsupported queries remain explicit catalogue
// entries so every candidate can be visited and audited.
struct TaskShapeCost {
  double predictedII = 0.0;
  double startupCycles = 0.0;
  bool supported = false;
};

struct RankedCandidate {
  std::string id;
  uint64_t manifestIndex = 0;
  double score = 0.0;
};

using CostQueryKey = std::tuple<std::string, int64_t, int64_t>;

// The task name is deliberately part of the catalogue key.  A task body SHA
// is useful for sharing an expensive prediction, but it is not a substitute
// for checking that every named task in the current IR has an explicit cost
// entry.
//
// PredictionCacheKey is the reusable prediction identity: the task body
// fingerprint binds the computation, the architecture fingerprint binds the
// target machine, and the oriented mapper dimensions bind the spatial query.
// A task's trip count is not included because the predictor returns the
// per-iteration II/startup pair; the scorer applies each task's current trip
// count afterward.
using PredictionCacheKey =
    std::tuple<std::string, std::string, int64_t, int64_t>;

// Loads one complete external predictor catalogue and memoizes lookups by the
// stable (task body, architecture, oriented mapper shape) identity. The
// per-task query table is retained separately so stale or extra catalogue
// records cannot be hidden by two tasks sharing a body.
class TaskShapeCostCache {
public:
  bool load(llvm::StringRef path, llvm::StringRef expectedFunction,
            llvm::ArrayRef<TaskMetadata> expectedTasks,
            // This is the SHA-256 of the exact candidate JSONL bytes that the
            // scorer will read.  It prevents a catalogue produced for another
            // enumeration (even one with the same task names) from being
            // reused accidentally.
            llvm::StringRef expectedCandidateManifestSha256,
            // This is the SHA-256 of the exact architecture YAML selected by
            // --architecture-spec.  Dimensions alone cannot identify a
            // machine: same-sized architectures may differ in routing,
            // functional units, memory, or latency data.
            llvm::StringRef expectedArchitectureSha256, std::string &error);
  const TaskShapeCost *get(const TaskShapeChoice &choice, std::string &error);

  llvm::StringRef nameSpace() const { return namespace_; }
  llvm::StringRef candidateManifestSha256() const {
    return candidateManifestSha256_;
  }
  llvm::StringRef architectureSha256() const { return architectureSha256_; }
  llvm::StringRef catalogSha256() const { return catalogSha256_; }
  llvm::StringRef mapperSuccessProbabilityRole() const {
    return mapperSuccessProbabilityRole_;
  }
  uint64_t hits() const { return hits_; }
  uint64_t misses() const { return misses_; }
  uint64_t cachedPredictions() const { return predictionCache_.size(); }
  uint64_t coveredQueries() const { return coveredQueries_.size(); }
  uint64_t catalogQueries() const { return catalog_.size(); }

private:
  std::string namespace_;
  // Identity of the candidate JSONL bound to this catalogue.  The score
  // header copies it so the Python driver can verify the complete chain.
  std::string candidateManifestSha256_;
  // Identity of the architecture YAML used by both enumeration and scoring.
  std::string architectureSha256_;
  // Identity of the exact cost JSON bytes loaded here.  This is reported in
  // the score header; it is not a hash of parsed/normalized JSON.
  std::string catalogSha256_;
  std::string mapperSuccessProbabilityRole_;
  // Current-IR task name -> task-body SHA-256.  This map is populated only
  // after the catalogue's task provenance has been checked against the IR.
  std::map<std::string, std::string> taskBodySha256_;
  std::map<CostQueryKey, TaskShapeCost> catalog_;
  std::map<PredictionCacheKey, TaskShapeCost> predictionCache_;
  std::set<CostQueryKey> coveredQueries_;
  uint64_t hits_ = 0;
  uint64_t misses_ = 0;
};

FailureOr<std::string> sha256File(llvm::StringRef path, std::string &error);
bool samePath(llvm::StringRef lhs, llvm::StringRef rhs);

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_TASK_COST_CATALOG_H

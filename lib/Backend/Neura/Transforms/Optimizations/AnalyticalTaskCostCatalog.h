//===- AnalyticalTaskCostCatalog.h ---------------------------*- C++ -*-===//
//
// Declares the validated task-shape ML cost oracle used by analytical DSE.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_TASK_COST_CATALOG_H
#define AMOEBA_ANALYTICAL_TASK_COST_CATALOG_H

#include "AnalyticalTaskCandidateSpace.h"

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
    "static-shape-compute-bottleneck";

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
using PredictionCacheKey =
    std::tuple<std::string, std::string, int64_t, int64_t>;

// Loads one complete external predictor catalogue and memoizes lookups by the
// stable (task body, architecture, oriented mapper shape) identity. The
// per-task query table is retained separately so stale or extra catalogue
// records cannot be hidden by two tasks sharing a body.
class TaskShapeCostCache {
public:
  bool load(llvm::StringRef path, llvm::StringRef expectedFunction,
            llvm::ArrayRef<TaskFact> expectedTasks,
            llvm::StringRef expectedCandidateManifestSha256,
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
  std::string candidateManifestSha256_;
  std::string architectureSha256_;
  std::string catalogSha256_;
  std::string mapperSuccessProbabilityRole_;
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

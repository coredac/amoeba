//===- AnalyticalTaskCostCatalog.cpp - Validated ML cost oracle ----------===//

#include "AnalyticalTaskCostCatalog.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SHA256.h"

#include <cmath>
#include <memory>
#include <optional>
#include <utility>

using namespace mlir;

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

static bool isSha256(StringRef value) {
  return value.size() == 64 && llvm::all_of(value, [](char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

static std::string sha256(StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

FailureOr<std::string> sha256File(StringRef path, std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error =
        "cannot fingerprint " + path.str() + ": " + buffer.getError().message();
    return failure();
  }
  return sha256((*buffer)->getBuffer());
}

static SmallString<256> normalizedPath(StringRef path) {
  SmallString<256> result(path);
  if (std::error_code ec = llvm::sys::fs::make_absolute(result))
    return SmallString<256>(path);
  llvm::sys::path::remove_dots(result, /*remove_dot_dot=*/true);
  return result;
}

bool samePath(StringRef lhs, StringRef rhs) {
  bool equivalent = false;
  if (!llvm::sys::fs::equivalent(lhs, rhs, equivalent) && equivalent)
    return true;
  return normalizedPath(lhs) == normalizedPath(rhs);
}

static std::optional<StringRef> requiredString(const llvm::json::Object &object,
                                               StringRef key,
                                               std::string &error) {
  std::optional<StringRef> value = object.getString(key);
  if (!value)
    error = "missing or invalid string field \"" + key.str() + "\"";
  return value;
}

static std::optional<int64_t> requiredInteger(const llvm::json::Object &object,
                                              StringRef key,
                                              std::string &error) {
  std::optional<int64_t> value = object.getInteger(key);
  if (!value)
    error = "missing or invalid integer field \"" + key.str() + "\"";
  return value;
}

static bool validateShaMap(const llvm::json::Object &object, StringRef label,
                           std::string &error) {
  for (const auto &entry : object) {
    std::optional<StringRef> value = entry.second.getAsString();
    if (entry.first.str().empty() || !value || !isSha256(*value)) {
      error = label.str() + " must map non-empty names to lowercase SHA-256";
      return false;
    }
  }
  return true;
}

static const llvm::json::Array *
supportedArchitectures(const llvm::json::Object &contract) {
  if (const llvm::json::Array *supported =
          contract.getArray("supported_architecture_sha256"))
    return supported;
  if (const llvm::json::Object *nested =
          contract.getObject("architecture_contract"))
    return nested->getArray("supported_architecture_sha256");
  return nullptr;
}

static bool sameCost(const TaskShapeCost &lhs, const TaskShapeCost &rhs) {
  return lhs.supported == rhs.supported &&
         (!lhs.supported || (lhs.predictedII == rhs.predictedII &&
                             lhs.startupCycles == rhs.startupCycles));
}

// Loads the catalogue only after validating all provenance required to bind it
// to the current IR, architecture, model metadata, and exact candidate bytes.
bool TaskShapeCostCache::load(StringRef path, StringRef expectedFunction,
                              ArrayRef<TaskFact> expectedTasks,
                              StringRef expectedCandidateManifestSha256,
                              StringRef expectedArchitectureSha256,
                              std::string &error) {
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot read cost catalogue " + path.str() + ": " +
            buffer.getError().message();
    return false;
  }
  catalogSha256_ = sha256((*buffer)->getBuffer());
  llvm::Expected<llvm::json::Value> parsed =
      llvm::json::parse((*buffer)->getBuffer());
  if (!parsed) {
    error =
        "invalid cost catalogue JSON: " + llvm::toString(parsed.takeError());
    return false;
  }
  llvm::json::Object *root = parsed->getAsObject();
  if (!root) {
    error = "cost catalogue must be a JSON object";
    return false;
  }
  auto schema = requiredString(*root, "schema", error);
  auto function = requiredString(*root, "function", error);
  auto modelNamespace = requiredString(*root, "namespace", error);
  llvm::json::Object *metadata = root->getObject("predictor_metadata");
  llvm::json::Array *entries = root->getArray("entries");
  if (!schema || !function || !modelNamespace || !metadata || !entries)
    return false;
  if (*schema != kCostSchema || *function != expectedFunction ||
      modelNamespace->empty()) {
    error = "cost catalogue schema/function/namespace mismatch";
    return false;
  }

  auto candidateSha =
      requiredString(*metadata, "candidate_manifest_sha256", error);
  llvm::json::Object *provenance = metadata->getObject("analytical_provenance");
  llvm::json::Object *architectureContract =
      metadata->getObject("architecture_contract");
  llvm::json::Object *rankingPolicy = metadata->getObject("ranking_policy");
  llvm::json::Object *model = metadata->getObject("model");
  if (!candidateSha || !provenance || !architectureContract || !rankingPolicy ||
      !model)
    return false;
  auto architectureSha =
      requiredString(*provenance, "architecture_sha256", error);
  auto neuraOptSha = requiredString(*provenance, "neura_opt_sha256", error);
  llvm::json::Object *taskBodyHashes =
      provenance->getObject("task_body_sha256");
  llvm::json::Object *taskHashes = provenance->getObject("task_dfg_sha256");
  if (!architectureSha || !neuraOptSha || !taskBodyHashes || !taskHashes)
    return false;
  if (!isSha256(*candidateSha) || !isSha256(*architectureSha) ||
      !isSha256(*neuraOptSha) ||
      !validateShaMap(*taskBodyHashes, "task body provenance", error) ||
      !validateShaMap(*taskHashes, "task DFG provenance", error)) {
    if (error.empty())
      error = "cost catalogue provenance contains an invalid SHA-256";
    return false;
  }
  if (*candidateSha != expectedCandidateManifestSha256) {
    error = "cost catalogue candidate manifest SHA-256 does not match the "
            "candidate file";
    return false;
  }
  if (*architectureSha != expectedArchitectureSha256) {
    error = "cost catalogue architecture SHA-256 does not match the current "
            "architecture";
    return false;
  }

  const llvm::json::Array *supported =
      supportedArchitectures(*architectureContract);
  if (!supported || supported->empty()) {
    error = "cost catalogue has no supported architecture contract";
    return false;
  }
  bool architectureIsSupported = false;
  for (const llvm::json::Value &value : *supported) {
    std::optional<StringRef> supportedSha = value.getAsString();
    if (!supportedSha || !isSha256(*supportedSha)) {
      error = "cost catalogue architecture contract contains an invalid "
              "SHA-256";
      return false;
    }
    architectureIsSupported |= *supportedSha == *architectureSha;
  }
  if (!architectureIsSupported) {
    error = "cost catalogue analytical architecture is outside the model's "
            "supported architecture contract";
    return false;
  }

  auto modelSha = requiredString(*model, "sha256", error);
  auto modelConfigSha = requiredString(*model, "config_sha256", error);
  if (!modelSha || !modelConfigSha || !isSha256(*modelSha) ||
      !isSha256(*modelConfigSha)) {
    error = "cost catalogue model or configuration SHA-256 is invalid";
    return false;
  }

  auto objective = requiredString(*rankingPolicy, "objective", error);
  auto probabilityRole =
      requiredString(*rankingPolicy, "mapper_success_probability", error);
  std::optional<bool> usesProbability =
      rankingPolicy->getBoolean("uses_mapper_success_probability");
  if (!objective || *objective != "predicted_compute_bottleneck" ||
      !probabilityRole ||
      (*probabilityRole != "diagnostic_only" &&
       *probabilityRole != "not_predicted") ||
      !usesProbability || *usesProbability) {
    error = "cost catalogue must exclude mapper success probability from "
            "support and ranking";
    return false;
  }

  if (taskBodyHashes->size() != expectedTasks.size() ||
      taskHashes->size() != expectedTasks.size()) {
    error = "cost catalogue task provenance does not exactly cover current IR "
            "tasks";
    return false;
  }
  taskBodySha256_.clear();
  for (const TaskFact &task : expectedTasks) {
    std::optional<StringRef> bodySha = taskBodyHashes->getString(task.name);
    std::optional<StringRef> dfgSha = taskHashes->getString(task.name);
    if (!bodySha || !dfgSha || *bodySha != task.bodySha256) {
      error = "cost catalogue task provenance does not bind the current task "
              "body to its source DFG";
      return false;
    }
    taskBodySha256_.emplace(task.name, task.bodySha256);
  }

  namespace_ = modelNamespace->str();
  candidateManifestSha256_ = candidateSha->str();
  architectureSha256_ = architectureSha->str();
  mapperSuccessProbabilityRole_ = probabilityRole->str();
  catalog_.clear();
  predictionCache_.clear();
  coveredQueries_.clear();
  hits_ = 0;
  misses_ = 0;
  std::map<PredictionCacheKey, TaskShapeCost> stablePredictions;
  for (llvm::json::Value &value : *entries) {
    llvm::json::Object *entry = value.getAsObject();
    if (!entry) {
      error = "cost entry is not an object";
      return false;
    }
    auto task = requiredString(*entry, "task", error);
    auto mapperRows = requiredInteger(*entry, "mapper_tile_rows", error);
    auto mapperCols = requiredInteger(*entry, "mapper_tile_cols", error);
    auto status = requiredString(*entry, "support_status", error);
    if (!task || !mapperRows || !mapperCols || !status)
      return false;
    auto body = taskBodySha256_.find(task->str());
    if (body == taskBodySha256_.end()) {
      error = "cost entry names a task outside current IR";
      return false;
    }
    if (*mapperRows <= 0 || *mapperCols <= 0) {
      error = "cost entry mapper tile dimensions must be positive";
      return false;
    }
    if (const llvm::json::Value *rawProbability =
            entry->get("mapper_success_probability")) {
      std::optional<double> probability = rawProbability->getAsNumber();
      if (*probabilityRole != "diagnostic_only" || !probability ||
          !std::isfinite(*probability) || *probability < 0.0 ||
          *probability > 1.0) {
        error = "mapper success probability must be a diagnostic in [0, 1]";
        return false;
      }
    }

    TaskShapeCost cost;
    if (*status == "supported") {
      auto ii = entry->getNumber("predicted_ii");
      auto startup = entry->getNumber("startup_cycles");
      auto lowerBound = entry->getNumber("analytical_lower_bound");
      if (!ii || !startup || !lowerBound || !std::isfinite(*ii) ||
          !std::isfinite(*startup) || !std::isfinite(*lowerBound) ||
          *ii <= 0.0 || *startup <= 0.0 || *lowerBound <= 0.0 ||
          *ii < *lowerBound) {
        error = "supported cost requires positive finite predicted_ii and "
                "startup_cycles and analytical_lower_bound, and "
                "predicted_ii >= analytical_lower_bound";
        return false;
      }
      cost = {*ii, *startup, true};
    } else if (*status != "unsupported") {
      error = "support_status must be supported or unsupported";
      return false;
    }
    CostQueryKey queryKey{task->str(), *mapperRows, *mapperCols};
    if (!catalog_.emplace(queryKey, cost).second) {
      error = "duplicate task/mapper-shape cost entry";
      return false;
    }
    PredictionCacheKey stableKey{body->second, architectureSha256_, *mapperRows,
                                 *mapperCols};
    auto [known, inserted] = stablePredictions.emplace(stableKey, cost);
    if (!inserted && !sameCost(known->second, cost)) {
      error = "tasks with the same body and mapper shape have conflicting "
              "predictions";
      return false;
    }
  }
  if (catalog_.empty()) {
    error = "cost catalogue contains no task-shape entries";
    return false;
  }
  return true;
}

const TaskShapeCost *TaskShapeCostCache::get(const TaskShapeChoice &choice,
                                             std::string &error) {
  CostQueryKey queryKey{choice.task, choice.shape.mapperRows,
                        choice.shape.mapperCols};
  auto found = catalog_.find(queryKey);
  if (found == catalog_.end()) {
    error = "missing cost for task=" + choice.task + ", mapper_shape=rect-" +
            std::to_string(choice.shape.mapperRows) + "x" +
            std::to_string(choice.shape.mapperCols);
    return nullptr;
  }
  coveredQueries_.insert(queryKey);
  auto body = taskBodySha256_.find(choice.task);
  if (body == taskBodySha256_.end()) {
    error = "cost lookup names a task outside current IR";
    return nullptr;
  }
  PredictionCacheKey stableKey{body->second, architectureSha256_,
                               choice.shape.mapperRows,
                               choice.shape.mapperCols};
  auto cached = predictionCache_.find(stableKey);
  if (cached != predictionCache_.end()) {
    ++hits_;
    return &cached->second;
  }
  ++misses_;
  auto inserted = predictionCache_.emplace(stableKey, found->second);
  return &inserted.first->second;
}

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

//===- ScoreAnalyticalTaskCandidatesPass.cpp -----------------------------===//
//
// Implements full-manifest scoring and deterministic top-k selection.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateManifest.h"
#include "AnalyticalTaskCostCatalog.h"

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/AnalyticalTaskDSE/SpatialDSEOrchestration.h"

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::analytical_dse;

namespace {

struct ScoreAnalyticalTaskCandidatesPass
    : public PassWrapper<ScoreAnalyticalTaskCandidatesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      ScoreAnalyticalTaskCandidatesPass)

  ScoreAnalyticalTaskCandidatesPass() = default;
  ScoreAnalyticalTaskCandidatesPass(
      const ScoreAnalyticalTaskCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "score-analytical-task-candidates";
  }
  StringRef getDescription() const override {
    return "Scores every frozen candidate through a shared task-shape ML "
           "cost cache, then emits deterministic top-k";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> candidateFile{
      *this, "candidates", llvm::cl::desc("Frozen candidate JSONL path."),
      llvm::cl::init("")};
  Option<std::string> costFile{
      *this, "cost-file", llvm::cl::desc("Task-shape cost catalogue path."),
      llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Score JSONL output path."),
                                 llvm::cl::init("")};
  Option<int64_t> topK{
      *this, "top-k",
      llvm::cl::desc("Uses zero to select every valid candidate."),
      llvm::cl::init(1)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    if (candidateFile.getValue().empty() || costFile.getValue().empty() ||
        outputFile.getValue().empty() || topK.getValue() < 0 ||
        samePath(candidateFile.getValue(), outputFile.getValue()) ||
        samePath(costFile.getValue(), outputFile.getValue()) ||
        samePath(candidateFile.getValue(), costFile.getValue())) {
      func.emitError() << "distinct candidates, cost-file, and output paths "
                          "and non-negative top-k are required";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskMetadata>> taskMetadata =
        collectAnalyticalTaskMetadata(func, error);
    if (failed(taskMetadata)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    // Bind scoring to the exact JSONL bytes that enumerate the candidate
    // space.  This includes the manifest header, candidate ordering, and
    // footer; a same-shaped space serialized differently is still a different
    // input to the cost catalogue.
    FailureOr<std::string> candidateSha =
        sha256File(candidateFile.getValue(), error);
    // Bind scoring to the exact architecture YAML selected for this pass.
    // The manifest and catalogue use this identity to reject costs generated
    // for a same-sized but otherwise different machine.
    FailureOr<std::string> architectureSha = currentArchitectureSha256(error);
    if (failed(candidateSha) || failed(architectureSha)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    TaskShapeCostCache costs;
    if (!costs.load(costFile.getValue(), func.getSymName(), *taskMetadata,
                    *candidateSha, *architectureSha, error)) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // Scores are held until readCandidateManifest has independently proven
    // canonical ordering, exact packing, and a complete footer.
    SmallVector<RankedCandidate> ranked;
    ManifestHeader manifestHeader;
    ManifestFooter manifestFooter;
    uint64_t scoredCount = 0;
    uint64_t validCount = 0;
    bool wrote = writeAtomically(
        outputFile.getValue(),
        [&](llvm::raw_ostream &os) {
          llvm::json::Object scoreHeader;
          scoreHeader["record_type"] = "header";
          scoreHeader["schema"] = kScoreSchema.str();
          scoreHeader["candidate_schema"] = kCandidateSchema.str();
          scoreHeader["function"] = func.getSymName().str();
          scoreHeader["cost_namespace"] = costs.nameSpace().str();
          // The Python driver checks these hashes before replaying any
          // materialized candidate.  They form a chain from the score output
          // back to the exact manifest, cost JSON, and architecture inputs.
          scoreHeader["candidate_manifest_sha256"] =
              costs.candidateManifestSha256().str();
          scoreHeader["cost_catalog_sha256"] = costs.catalogSha256().str();
          scoreHeader["architecture_sha256"] = costs.architectureSha256().str();
          scoreHeader["score_model"] = kScoreModel.str();
          scoreHeader["mapper_success_probability"] =
              costs.mapperSuccessProbabilityRole().str();
          writeJsonLine(os, std::move(scoreHeader));

          auto consume = [&](uint64_t manifestIndex, const Candidate &candidate,
                             std::string &consumeError) {
            // duration(task, shape) = startup + II * (trip_count - 1)
            // score(candidate)      = production scheduler makespan
            bool valid = true;
            double bottleneck = 0.0;
            SmallVector<int64_t> schedulerDurations;
            std::string rejectReason;
            llvm::json::Array taskCosts;
            for (const TaskShapeChoice &choice : candidate.choices) {
              const TaskShapeCost *cost = costs.get(choice, consumeError);
              if (!cost)
                return false;
              llvm::json::Object taskCost;
              taskCost["task"] = choice.task;
              taskCost["mapper_tile_rows"] = choice.shape.mapperRows;
              taskCost["mapper_tile_cols"] = choice.shape.mapperCols;
              taskCost["trip_count"] = choice.tripCount;
              if (!cost->supported) {
                valid = false;
                if (rejectReason.empty())
                  rejectReason = "UNSUPPORTED_TASK_SHAPE";
                taskCost["support_status"] = "unsupported";
              } else {
                taskCost["predicted_ii"] = cost->predictedII;
                taskCost["startup_cycles"] = cost->startupCycles;
                double duration = cost->startupCycles +
                                  cost->predictedII *
                                      static_cast<double>(choice.tripCount - 1);
                if (!std::isfinite(duration)) {
                  consumeError =
                      "task duration overflow for task=" + choice.task +
                      ", mapper_shape=rect-" +
                      std::to_string(choice.shape.mapperRows) + "x" +
                      std::to_string(choice.shape.mapperCols);
                  return false;
                }
                taskCost["support_status"] = "supported";
                taskCost["predicted_duration"] = duration;
                bottleneck = std::max(bottleneck, duration);
                double scheduledDuration = std::ceil(duration);
                if (scheduledDuration >
                    static_cast<double>(std::numeric_limits<int64_t>::max())) {
                  consumeError =
                      "task duration exceeds scheduler range for task=" +
                      choice.task;
                  return false;
                }
                schedulerDurations.push_back(std::max<int64_t>(
                    1, static_cast<int64_t>(scheduledDuration)));
              }
              taskCosts.push_back(std::move(taskCost));
            }

            llvm::json::Object score;
            score["record_type"] = "score";
            score["schema"] = kScoreSchema.str();
            score["candidate_id"] = candidate.id;
            score["valid"] = valid;
            score["task_costs"] = std::move(taskCosts);
            if (valid) {
              score["predicted_compute_bottleneck"] = bottleneck;
              // The clone receives the same fixed shapes and task priority as
              // final orchestration. Only its predicted durations differ.
              OwningOpRef<ModuleOp> clonedModule = module.clone();
              FailureOr<func::FuncOp> clonedFunction = selectTaskFunction(
                  *clonedModule, func.getSymName(), consumeError);
              if (failed(clonedFunction))
                return false;
              SmallVector<taskflow::TaskflowTaskOp> clonedTasks;
              clonedFunction->walk([&](taskflow::TaskflowTaskOp task) {
                clonedTasks.push_back(task);
              });
              if (clonedTasks.size() != candidate.choices.size() ||
                  clonedTasks.size() != schedulerDurations.size()) {
                consumeError = "candidate task count changed while cloning";
                return false;
              }
              OpBuilder builder(func.getContext());
              for (auto [task, choice, duration] : llvm::zip_equal(
                       clonedTasks, candidate.choices, schedulerDurations)) {
                task->setAttr("cgra_count", builder.getI32IntegerAttr(
                                                choice.shape.cgraCount()));
                task->setAttr(
                    "cgra_shape",
                    builder.getStringAttr(choice.shape.toCgraShapeAttrValue()));
                task->setAttr("amoeba.analytical_shape_orientation_fixed",
                              builder.getUnitAttr());
                task->setAttr("est_latency",
                              builder.getI64IntegerAttr(duration));
              }
              taskflow::SpatialDSEOrchestration strategy(
                  static_cast<int>(manifestHeader.gridRows),
                  static_cast<int>(manifestHeader.gridCols),
                  taskflow::SchedulingMode::SpatialTemporal);
              taskflow::TaskScheduler scheduler(
                  static_cast<int>(manifestHeader.gridRows),
                  static_cast<int>(manifestHeader.gridCols),
                  taskflow::SchedulingMode::SpatialTemporal,
                  taskflow::ShapeSelectionPolicy::FixedOrientation);
              if (!scheduler.schedule(
                      *clonedFunction,
                      strategy.computeRoutingCriticalPathPriority(
                          *clonedFunction))) {
                consumeError =
                    "spatial-temporal scheduler rejected candidate " +
                    candidate.id;
                return false;
              }
              int64_t makespan = scheduler.getScheduleMakespan();
              if (makespan <= 0) {
                consumeError = "scheduler produced an empty or overflowing "
                               "makespan for " +
                               candidate.id;
                return false;
              }
              score["predicted_scheduler_makespan"] = makespan;
              ranked.push_back(
                  {candidate.id, manifestIndex, static_cast<double>(makespan)});
              ++validCount;
            } else {
              score["reject_reason"] = rejectReason;
            }
            writeJsonLine(os, std::move(score));
            ++scoredCount;
            return true;
          };

          if (!readCandidateManifest(candidateFile.getValue(), *taskMetadata,
                                     func.getSymName(),
                                     ::mlir::neura::getArchitecture(), consume,
                                     manifestHeader, manifestFooter, error))
            return false;
          if (manifestHeader.architectureSha256 != costs.architectureSha256() ||
              scoredCount != manifestFooter.candidateCount) {
            error = "not every frozen candidate was scored under the bound "
                    "architecture";
            return false;
          }
          if (costs.coveredQueries() != costs.catalogQueries()) {
            error = "cost catalogue does not exactly cover candidate manifest "
                    "task-shape queries";
            return false;
          }

          // Sort only after complete traversal. Numeric manifest order is the
          // deterministic tie-break, avoiding candidate-13 < candidate-5.
          llvm::sort(ranked, [](const RankedCandidate &lhs,
                                const RankedCandidate &rhs) {
            if (lhs.score != rhs.score)
              return lhs.score < rhs.score;
            return lhs.manifestIndex < rhs.manifestIndex;
          });
          uint64_t selected =
              topK.getValue() == 0
                  ? ranked.size()
                  : std::min<uint64_t>(topK.getValue(), ranked.size());
          llvm::json::Array shortlist;
          for (uint64_t index = 0; index < selected; ++index) {
            llvm::json::Object item;
            item["rank"] = static_cast<int64_t>(index);
            item["candidate_id"] = ranked[index].id;
            item["predicted_scheduler_makespan"] = ranked[index].score;
            shortlist.push_back(std::move(item));
          }

          llvm::json::Object cacheStats;
          cacheStats["hits"] = static_cast<int64_t>(costs.hits());
          cacheStats["misses"] = static_cast<int64_t>(costs.misses());
          cacheStats["predictions"] =
              static_cast<int64_t>(costs.cachedPredictions());
          cacheStats["queries"] = static_cast<int64_t>(costs.coveredQueries());
          llvm::json::Object footer;
          footer["record_type"] = "footer";
          footer["schema"] = kScoreSchema.str();
          footer["candidate_count"] =
              static_cast<int64_t>(manifestFooter.candidateCount);
          footer["scored_count"] = static_cast<int64_t>(scoredCount);
          footer["valid_count"] = static_cast<int64_t>(validCount);
          footer["top_k_requested"] = topK.getValue();
          footer["shortlist"] = std::move(shortlist);
          footer["cache"] = std::move(cacheStats);
          writeJsonLine(os, std::move(footer));
          return true;
        },
        error);
    if (!wrote) {
      func.emitError() << error;
      return signalPassFailure();
    }
    llvm::errs() << "[AnalyticalTaskDSE] scored all " << scoredCount
                 << " candidates with " << costs.cachedPredictions()
                 << " cached task-body/shape predictions\n";
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createScoreAnalyticalTaskCandidatesPass() {
  return std::make_unique<ScoreAnalyticalTaskCandidatesPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir

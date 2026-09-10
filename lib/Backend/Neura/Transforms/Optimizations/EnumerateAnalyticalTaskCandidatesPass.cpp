//===- EnumerateAnalyticalTaskCandidatesPass.cpp -------------------------===//
//
// Implements the pass that freezes every concurrently packable rectangular
// task-shape tuple.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateSpace.h"

#include "Backend/Neura/NeuraBackendPasses.h"

#include "NeuraDialect/Architecture/Architecture.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/JSON.h"

#include <cstdint>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::analytical_dse;

namespace {

struct EnumerateAnalyticalTaskCandidatesPass
    : public PassWrapper<EnumerateAnalyticalTaskCandidatesPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      EnumerateAnalyticalTaskCandidatesPass)

  EnumerateAnalyticalTaskCandidatesPass() = default;
  EnumerateAnalyticalTaskCandidatesPass(
      const EnumerateAnalyticalTaskCandidatesPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "enumerate-analytical-task-candidates";
  }
  StringRef getDescription() const override {
    return "Freezes every rectangular task-shape candidate without scoring or "
           "running the mapper";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> outputFile{*this, "output",
                                 llvm::cl::desc("Candidate JSONL output path."),
                                 llvm::cl::init("")};
  void runOnOperation() override {
    // Selects the Taskflow function. Besides the candidate file, a successful
    // run attaches each canonical task-body hash to its source task so derived
    // artifacts can bind to the same body.
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selectedFunction =
        selectTaskFunction(module, functionName.getValue(), error);
    if (failed(selectedFunction)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selectedFunction;
    if (outputFile.getValue().empty()) {
      func.emitError() << "output is required";
      return signalPassFailure();
    }

    // Collects task facts and builds the single-task shape alphabet from the
    // architecture values read by Neura's YAML loader. Shape enumeration does
    // not require a numeric trip count, so symbol-dynamic tasks remain valid.
    FailureOr<SmallVector<TaskFact>> taskFacts =
        collectAnalyticalTaskFacts(func, error);
    if (failed(taskFacts)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    SmallVector<RectShape> shapes = enumerateStaticRectShapes(
        architecture.getMultiCgraRows(), architecture.getMultiCgraColumns(),
        architecture.getPerCgraRows(), architecture.getPerCgraColumns());
    if (shapes.empty()) {
      func.emitError() << "declared rectangular shape space is empty";
      return signalPassFailure();
    }
    FailureOr<std::string> architectureSha = currentArchitectureSha256(error);
    if (failed(architectureSha)) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // Counts the complete *feasible* shape space before publishing anything.
    // A tuple is feasible only when its fixed-orientation task rectangles have
    // an exact simultaneous, non-overlapping placement on the physical grid.
    // The concrete origins remain a downstream heuristic choice; temporal
    // reuse cannot rescue an over-capacity tuple in this search scope.
    uint64_t candidateCount = 0;
    ConcurrentPackingCache packing(architecture.getMultiCgraRows(),
                                   architecture.getMultiCgraColumns());
    SmallVector<SmallVector<uint8_t>> usedCostQueries(taskFacts->size());
    for (SmallVector<uint8_t> &used : usedCostQueries)
      used.assign(shapes.size(), 0);
    bool countedAll = visitConcurrentlyPackableShapeTuples(
        taskFacts->size(), shapes, packing,
        [&](uint64_t index, ArrayRef<size_t> shapeIndices) {
          candidateCount = index + 1;
          for (auto [taskIndex, shapeIndex] : llvm::enumerate(shapeIndices))
            usedCostQueries[taskIndex][shapeIndex] = 1;
          return true;
        });
    if (!countedAll) {
      func.emitError() << "failed while counting the packable shape space";
      return signalPassFailure();
    }
    if (candidateCount == 0) {
      func.emitError() << "no task shape tuple can fit simultaneously on the "
                          "physical CGRA grid";
      return signalPassFailure();
    }

    const std::string function = func.getSymName().str();
    bool wrote = writeAtomically(
        outputFile.getValue(),
        [&](llvm::raw_ostream &os) {
          // Freezes every input needed to reconstruct the candidate space. The
          // cost-query list contains exactly the task/shape pairs referenced by
          // at least one feasible candidate.
          llvm::json::Object architectureRecord;
          architectureRecord["grid_rows"] =
              int64_t{architecture.getMultiCgraRows()};
          architectureRecord["grid_cols"] =
              int64_t{architecture.getMultiCgraColumns()};
          architectureRecord["per_cgra_tile_rows"] =
              int64_t{architecture.getPerCgraRows()};
          architectureRecord["per_cgra_tile_cols"] =
              int64_t{architecture.getPerCgraColumns()};
          architectureRecord["spec_sha256"] = *architectureSha;
          llvm::json::Array tasks;
          for (const TaskFact &task : *taskFacts) {
            llvm::json::Object record;
            record["task"] = task.name;
            record["body_sha256"] = task.bodySha256;
            if (task.tripCount)
              record["trip_count"] = *task.tripCount;
            else
              record["trip_count_kind"] = kSymbolDynamicTripCountKind.str();
            tasks.push_back(std::move(record));
          }
          llvm::json::Array costQueries;
          for (auto [taskIndex, task] : llvm::enumerate(*taskFacts)) {
            for (auto [shapeIndex, shape] : llvm::enumerate(shapes)) {
              if (!usedCostQueries[taskIndex][shapeIndex])
                continue;
              llvm::json::Object query;
              query["task"] = task.name;
              query["mapper_tile_rows"] = shape.mapperRows;
              query["mapper_tile_cols"] = shape.mapperCols;
              costQueries.push_back(std::move(query));
            }
          }
          // Records the axes held constant by this static shape-selection
          // contract.
          llvm::json::Object fixedAxes;
          fixedAxes["fusion"] = "identity";
          fixedAxes["fission"] = "factor-1";
          fixedAxes["tiling"] = "factor-1";
          fixedAxes["placement"] =
              "exact-fit-required-coordinates-downstream-heuristic";
          fixedAxes["temporal_order"] = "downstream-heuristic";
          fixedAxes["communication"] = "not-scored";
          llvm::json::Object header;
          header["record_type"] = "header";
          header["schema"] = kCandidateSchema.str();
          header["search_scope"] = kSearchScope.str();
          header["shape_policy"] = kShapePolicy.str();
          header["spatial_capacity_policy"] = kSpatialCapacityPolicy.str();
          header["function"] = function;
          header["architecture"] = std::move(architectureRecord);
          header["tasks"] = std::move(tasks);
          header["cost_queries"] = std::move(costQueries);
          header["fixed_axes"] = std::move(fixedAxes);
          writeJsonLine(os, std::move(header));

          // Emits every concurrently packable tuple in task-major shape order.
          // A tuple is emitted once even if it has multiple legal placements;
          // placement itself is not a DSE axis yet.
          uint64_t emitted = 0;
          bool emittedAll = visitConcurrentlyPackableShapeTuples(
              taskFacts->size(), shapes, packing,
              [&](uint64_t index, ArrayRef<size_t> shapeIndices) {
                Candidate candidate;
                candidate.id = makeSequentialCandidateId(index);
                for (auto [taskIndex, shapeIndex] :
                     llvm::enumerate(shapeIndices)) {
                  const TaskFact &task = (*taskFacts)[taskIndex];
                  candidate.choices.push_back(
                      {task.name, task.tripCount, shapes[shapeIndex]});
                }
                writeJsonLine(os, candidateJson(candidate));
                emitted = index + 1;
                return true;
              });
          if (!emittedAll || emitted != candidateCount) {
            error = "internal candidate-count mismatch";
            return false;
          }

          // Closes the stream with its record count. The common reader
          // independently recomputes the exact packable space and every ID.
          llvm::json::Object footer;
          footer["record_type"] = "footer";
          footer["schema"] = kCandidateSchema.str();
          footer["candidate_count"] = static_cast<int64_t>(emitted);
          writeJsonLine(os, std::move(footer));
          return true;
        },
        error);
    if (!wrote) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // Publish the binding only after the complete manifest has been written.
    // A failed or truncated enumeration therefore cannot leave IR that looks
    // paired with a usable candidate file. The hash routine deliberately
    // ignores this attribute, so re-enumerating this output is idempotent.
    for (const TaskFact &task : *taskFacts)
      task.op->setAttr(kSourceTaskBodyShaAttr,
                       StringAttr::get(func.getContext(), task.bodySha256));
    llvm::errs() << "[AnalyticalTaskDSE] enumerated all " << candidateCount
                 << " concurrently packable shape candidates into "
                 << outputFile.getValue() << "\n";
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createEnumerateAnalyticalTaskCandidatesPass() {
  return std::make_unique<EnumerateAnalyticalTaskCandidatesPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir

//===- EnumerateAnalyticalTaskCandidatesPass.cpp -------------------------===//
//
// Implements the pass that freezes every concurrently packable rectangular
// task-shape tuple.
//
//===----------------------------------------------------------------------===//

#include "SpatialTaskCandidateSpace.h"

#include "Backend/Neura/NeuraBackendPasses.h"

#include "NeuraDialect/Architecture/Architecture.h"

#include "mlir/Pass/Pass.h"

#include "llvm/Support/JSON.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::analytical_candidates;

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

  Option<std::string> function_name{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> output_file{
      *this, "output", llvm::cl::desc("Candidate JSONL output path."),
      llvm::cl::init("")};
  Option<int64_t> max_candidates{
      *this, "max-candidates",
      llvm::cl::desc("Fails rather than publishing a partial manifest."),
      llvm::cl::init(1000000)};
  Option<int64_t> max_cgras_per_task{
      *this, "max-cgras-per-task",
      llvm::cl::desc(
          "Limits each task footprint; zero uses the full physical grid."),
      llvm::cl::init(0)};

  void runOnOperation() override {
    // Selects the Taskflow function. Besides the candidate file, a successful
    // run attaches each canonical task-body hash to its source task so derived
    // artifacts can bind to the same body.
    ModuleOp module = getOperation();
    std::string error;
    FailureOr<func::FuncOp> selected_function =
        selectTaskFunction(module, function_name.getValue(), error);
    if (failed(selected_function)) {
      module.emitError() << error;
      return signalPassFailure();
    }
    func::FuncOp func = *selected_function;
    if (output_file.getValue().empty() || max_candidates.getValue() <= 0 ||
        max_cgras_per_task.getValue() < 0) {
      func.emitError() << "output, positive max-candidates, and nonnegative "
                          "max-cgras-per-task are required";
      return signalPassFailure();
    }

    // Collects task metadata and builds the single-task shape alphabet from the
    // architecture values read by Neura's YAML loader. Dynamic or unresolved
    // trip counts are rejected by the static candidate-space contract.
    FailureOr<SmallVector<TaskMetadata>> task_metadata =
        collectAnalyticalTaskMetadata(func, error);
    if (failed(task_metadata)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    const ::mlir::neura::Architecture &architecture =
        ::mlir::neura::getArchitecture();
    const int64_t grid_rows = architecture.getMultiCgraRows();
    const int64_t grid_cols = architecture.getMultiCgraColumns();
    if (grid_rows <= 0 || grid_cols <= 0 ||
        grid_rows > std::numeric_limits<int64_t>::max() / grid_cols) {
      func.emitError() << "physical CGRA grid dimensions are invalid";
      return signalPassFailure();
    }
    const int64_t grid_area = grid_rows * grid_cols;
    const int64_t effective_max_cgras_per_task =
        max_cgras_per_task.getValue() == 0
            ? grid_area
            : std::min(max_cgras_per_task.getValue(), grid_area);
    SmallVector<RectShape> shapes = enumerateStaticRectShapes(
        grid_rows, grid_cols, architecture.getPerCgraRows(),
        architecture.getPerCgraColumns(), effective_max_cgras_per_task);
    if (shapes.empty()) {
      func.emitError() << "declared rectangular shape space is empty";
      return signalPassFailure();
    }
    // TODO: Introduce an explicit, validated shape-pruning policy after the
    // complete rectangular space has a stable downstream contract.
    SmallVector<SmallVector<RectShape>> shapes_by_task(task_metadata->size(),
                                                       shapes);
    FailureOr<std::string> architecture_sha = currentArchitectureSha256(error);
    if (failed(architecture_sha)) {
      func.emitError() << error;
      return signalPassFailure();
    }

    // Counts the complete *feasible* shape space before publishing anything.
    // A tuple is feasible only when its fixed-rotation task rectangles have
    // an exact simultaneous, non-overlapping placement on the physical grid.
    // The concrete origins remain a downstream heuristic choice; temporal
    // reuse cannot rescue an over-capacity tuple in this search scope.
    uint64_t candidate_count = 0;
    bool exceeded_limit = false;
    ConcurrentPackingCache packing(grid_rows, grid_cols);
    SmallVector<SmallVector<uint8_t>> used_cost_queries(task_metadata->size());
    for (auto [task_index, used] : llvm::enumerate(used_cost_queries)) {
      used.assign(shapes_by_task[task_index].size(), 0);
    }
    bool counted_all = visitConcurrentlyPackableShapeTuples(
        shapes_by_task, packing,
        [&](uint64_t index, ArrayRef<size_t> shape_indices) {
          if (index >= static_cast<uint64_t>(max_candidates.getValue())) {
            exceeded_limit = true;
            return false;
          }
          candidate_count = index + 1;
          for (auto [task_index, shape_index] :
               llvm::enumerate(shape_indices)) {
            used_cost_queries[task_index][shape_index] = 1;
          }
          return true;
        });
    if (!counted_all && exceeded_limit) {
      func.emitError() << "complete concurrently packable shape space exceeds "
                          "max-candidates="
                       << max_candidates.getValue()
                       << "; refusing to publish a partial candidate manifest";
      return signalPassFailure();
    }
    if (!counted_all) {
      func.emitError() << "failed while counting the packable shape space";
      return signalPassFailure();
    }
    if (candidate_count == 0) {
      func.emitError() << "no task shape tuple can fit simultaneously on the "
                          "physical CGRA grid";
      return signalPassFailure();
    }

    const std::string function = func.getSymName().str();
    bool wrote = writeAtomically(
        output_file.getValue(),
        [&](llvm::raw_ostream &os) {
          // Freezes every input needed to reconstruct the candidate space. The
          // architecture YAML hash and each source-task body hash bind this
          // manifest to the exact machine and computation that the predictor
          // and materializer consume. The cost-query list contains exactly
          // the task/shape pairs referenced by at least one feasible
          // candidate.
          llvm::json::Object architecture_record;
          architecture_record["grid_rows"] =
              int64_t{architecture.getMultiCgraRows()};
          architecture_record["grid_cols"] =
              int64_t{architecture.getMultiCgraColumns()};
          architecture_record["per_cgra_tile_rows"] =
              int64_t{architecture.getPerCgraRows()};
          architecture_record["per_cgra_tile_cols"] =
              int64_t{architecture.getPerCgraColumns()};
          architecture_record["spec_sha256"] = *architecture_sha;
          llvm::json::Array tasks;
          for (const TaskMetadata &task : *task_metadata) {
            llvm::json::Object record;
            record["task"] = task.name;
            record["body_sha256"] = task.body_sha256;
            record["trip_count"] = task.trip_count;
            tasks.push_back(std::move(record));
          }
          llvm::json::Array cost_queries;
          for (auto [task_index, task] : llvm::enumerate(*task_metadata)) {
            for (auto [shape_index, shape] :
                 llvm::enumerate(shapes_by_task[task_index])) {
              if (!used_cost_queries[task_index][shape_index]) {
                continue;
              }
              llvm::json::Object query;
              query["task"] = task.name;
              query["mapper_tile_rows"] = shape.mapper_rows;
              query["mapper_tile_cols"] = shape.mapper_cols;
              cost_queries.push_back(std::move(query));
            }
          }
          // Records the axes held constant by this static shape-selection
          // contract.
          llvm::json::Object fixed_axes;
          fixed_axes["fusion"] = "identity";
          fixed_axes["fission"] = "factor-1";
          fixed_axes["tiling"] = "factor-1";
          fixed_axes["placement"] =
              "exact-fit-required-coordinates-downstream-heuristic";
          fixed_axes["temporal_order"] = "downstream-heuristic";
          fixed_axes["communication"] = "not-scored";
          llvm::json::Object header;
          header["record_type"] = "header";
          header["schema"] = kCandidateSchema.str();
          header["search_scope"] = kSearchScope.str();
          header["shape_policy"] = kShapePolicy.str();
          header["spatial_capacity_policy"] = kSpatialCapacityPolicy.str();
          header["shape_pruning_policy"] = kShapePruningPolicy.str();
          header["function"] = function;
          header["architecture"] = std::move(architecture_record);
          header["max_cgras_per_task"] = effective_max_cgras_per_task;
          header["tasks"] = std::move(tasks);
          header["cost_queries"] = std::move(cost_queries);
          header["fixed_axes"] = std::move(fixed_axes);
          writeJsonLine(os, std::move(header));

          // Emits every concurrently packable tuple in task-major shape order.
          // A tuple is emitted once even if it has multiple legal placements;
          // concrete placement is not part of this candidate space.
          uint64_t emitted = 0;
          bool emitted_all = visitConcurrentlyPackableShapeTuples(
              shapes_by_task, packing,
              [&](uint64_t index, ArrayRef<size_t> shape_indices) {
                Candidate candidate;
                candidate.id = makeSequentialCandidateId(index);
                for (auto [task_index, shape_index] :
                     llvm::enumerate(shape_indices)) {
                  const TaskMetadata &task = (*task_metadata)[task_index];
                  candidate.choices.push_back(
                      {task.name, task.trip_count,
                       shapes_by_task[task_index][shape_index]});
                }
                writeJsonLine(os, candidateJson(candidate));
                emitted = index + 1;
                return true;
              });
          if (!emitted_all || emitted != candidate_count) {
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

    // Publish the source-task binding only after the complete manifest has
    // been written. A failed or truncated enumeration therefore cannot leave
    // IR that looks paired with a usable candidate file. The body-hash routine
    // deliberately ignores this attribute, so re-enumerating this output is
    // idempotent while any real task-body edit still invalidates the manifest.
    for (const TaskMetadata &task : *task_metadata) {
      task.op->setAttr(kSourceTaskBodyShaAttr,
                       StringAttr::get(func.getContext(), task.body_sha256));
    }
    llvm::errs() << "[AnalyticalTaskCandidates] enumerated all "
                 << candidate_count
                 << " concurrently packable shape candidates into "
                 << output_file.getValue() << "\n";
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

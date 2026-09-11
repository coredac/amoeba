//===- MaterializeAnalyticalTaskCandidatePass.cpp ------------------------===//
//
// Implements replay of one validated shape candidate onto Taskflow IR.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateManifest.h"

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/orchestration_utils.h"

#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <optional>
#include <string>

using namespace mlir;
using namespace mlir::amoeba::neura::analytical_dse;

namespace {

struct MaterializeAnalyticalTaskCandidatePass
    : public PassWrapper<MaterializeAnalyticalTaskCandidatePass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      MaterializeAnalyticalTaskCandidatePass)

  MaterializeAnalyticalTaskCandidatePass() = default;
  MaterializeAnalyticalTaskCandidatePass(
      const MaterializeAnalyticalTaskCandidatePass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "materialize-analytical-task-candidate";
  }
  StringRef getDescription() const override {
    return "Writes one selected shape candidate onto Taskflow IR without "
           "invoking the mapper";
  }

  Option<std::string> functionName{
      *this, "function",
      llvm::cl::desc("Taskflow function; inferred when exactly one exists."),
      llvm::cl::init("")};
  Option<std::string> candidateFile{
      *this, "candidates", llvm::cl::desc("Frozen candidate JSONL path."),
      llvm::cl::init("")};
  Option<std::string> candidateIdOption{
      *this, "candidate-id", llvm::cl::desc("Candidate ID to materialize."),
      llvm::cl::init("")};

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
    if (candidateFile.getValue().empty() ||
        candidateIdOption.getValue().empty()) {
      func.emitError() << "candidates and candidate-id are required";
      return signalPassFailure();
    }

    FailureOr<SmallVector<TaskFact>> taskFacts =
        collectAnalyticalTaskFacts(func, error);
    if (failed(taskFacts)) {
      func.emitError() << error;
      return signalPassFailure();
    }
    // Continues through the footer after finding the requested record. This
    // validates the complete manifest and detects any second match.
    std::optional<Candidate> selected;
    ManifestHeader header;
    ManifestFooter footer;
    auto consume = [&](uint64_t, const Candidate &candidate, std::string &) {
      if (candidate.id == candidateIdOption.getValue()) {
        if (selected) {
          return false;
        }
        selected = candidate;
      }
      return true;
    };
    if (!readCandidateManifest(
            candidateFile.getValue(), *taskFacts, func.getSymName(),
            ::mlir::neura::getArchitecture(), consume, header, footer, error)) {
      if (error.empty()) {
        error = "candidate selection is ambiguous";
      }
      func.emitError() << error;
      return signalPassFailure();
    }
    if (!selected) {
      func.emitError() << "requested candidate is absent from the complete "
                          "manifest";
      return signalPassFailure();
    }

    // Delays mutation until the manifest, IR, architecture, and record have
    // all been validated. These attributes configure the unchanged downstream
    // heuristic mapper; this pass fabricates no placement or II.
    OpBuilder builder(func.getContext());
    for (auto [task, choice] : llvm::zip(*taskFacts, selected->choices)) {
      task.op->setAttr("cgra_count",
                       builder.getI32IntegerAttr(
                           static_cast<int32_t>(choice.shape.cgraCount())));
      task.op->setAttr("cgra_shape", builder.getStringAttr(
                                         choice.shape.toCgraShapeAttrValue()));
      // The rectangle orientation is part of the selected candidate and must
      // remain unchanged during resource allocation.
      task.op->setAttr("amoeba.analytical_shape_orientation_fixed",
                       builder.getUnitAttr());
    }
    func->setAttr("analytical_task_candidate_id",
                  builder.getStringAttr(selected->id));
    func->setAttr("analytical_task_candidate_scope",
                  builder.getStringAttr(kSearchScope));
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createMaterializeAnalyticalTaskCandidatePass() {
  return std::make_unique<MaterializeAnalyticalTaskCandidatePass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir

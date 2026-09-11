//===- AnalyzeAnalyticalLowerBoundPass.cpp - Task-shape bounds -----------===//
//
// Exports the mapper's analytical lower-bound components for the external
// predictor adapter. It never performs placement or routing.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/NeuraBackendPasses.h"

#include "NeuraDialect/Architecture/Architecture.h"
#include "NeuraDialect/Mapping/mapping_util.h"
#include "NeuraDialect/NeuraOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>

using namespace mlir;

namespace {

struct AnalyzeAnalyticalLowerBoundPass
    : public PassWrapper<AnalyzeAnalyticalLowerBoundPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
      AnalyzeAnalyticalLowerBoundPass)

  AnalyzeAnalyticalLowerBoundPass() = default;
  AnalyzeAnalyticalLowerBoundPass(
      const AnalyzeAnalyticalLowerBoundPass &other)
      : PassWrapper(other) {}

  StringRef getArgument() const override {
    return "analyze-analytical-lower-bound";
  }
  StringRef getDescription() const override {
    return "Computes the mapper-compatible analytical lower bound without "
           "mapping";
  }

  Option<int64_t> xTiles{
      *this, "x-tiles",
      llvm::cl::desc("Concrete mapper tile columns; must be positive."),
      llvm::cl::init(0)};
  Option<int64_t> yTiles{
      *this, "y-tiles",
      llvm::cl::desc("Concrete mapper tile rows; must be positive."),
      llvm::cl::init(0)};

  void runOnOperation() override {
    ModuleOp module = getOperation();
    if (xTiles.getValue() <= 0 || yTiles.getValue() <= 0 ||
        xTiles.getValue() > std::numeric_limits<int>::max() ||
        yTiles.getValue() > std::numeric_limits<int>::max()) {
      module.emitError()
          << "analyze-analytical-lower-bound requires positive x-tiles and "
             "y-tiles within the integer range";
      return signalPassFailure();
    }

    // Extracted task DFGs contain one func.func. A hand-written fixture may
    // retain one neura.kernel instead, but multiple regions are ambiguous.
    llvm::SmallVector<Region *> regions;
    module.walk([&](::mlir::neura::KernelOp kernel) {
      if (!kernel.getBody().empty())
        regions.push_back(&kernel.getBody());
    });
    if (regions.empty()) {
      module.walk([&](func::FuncOp func) {
        if (!func.getBody().empty())
          regions.push_back(&func.getBody());
      });
    }
    if (regions.size() != 1) {
      module.emitError() << "analyze-analytical-lower-bound expects exactly "
                            "one standalone task DFG";
      return signalPassFailure();
    }

    const ::mlir::neura::Architecture &base = ::mlir::neura::getArchitecture();
    std::unique_ptr<::mlir::neura::Architecture> shape =
        base.cloneWithNewDimensions(static_cast<int>(yTiles.getValue()),
                                    static_cast<int>(xTiles.getValue()));
    if (!shape || shape->getNumTiles() <= 0) {
      module.emitError() << "cannot construct the requested mapper shape";
      return signalPassFailure();
    }

    llvm::SmallVector<::mlir::neura::RecurrenceCycle, 4> recurrenceCycles =
        ::mlir::neura::collectRecurrenceCycles(*regions.front());
    int recMii = 1;
    for (const auto &cycle : recurrenceCycles)
      recMii = std::max(recMii, cycle.length);
    int resMii = ::mlir::neura::calculateResMii(*regions.front(), *shape);
    int analyticalLowerBound = std::max(recMii, resMii);

    MLIRContext *context = module.getContext();
    module->setAttr("amoeba.analytical_lower_bound_info",
                    UnitAttr::get(context));
    module->setAttr("amoeba.analytical_lower_bound",
                    IntegerAttr::get(IntegerType::get(context, 32),
                                     analyticalLowerBound));
    module->setAttr("amoeba.rec_mii",
                    IntegerAttr::get(IntegerType::get(context, 32), recMii));
    module->setAttr("amoeba.res_mii",
                    IntegerAttr::get(IntegerType::get(context, 32), resMii));
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createAnalyzeAnalyticalLowerBoundPass() {
  return std::make_unique<AnalyzeAnalyticalLowerBoundPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir

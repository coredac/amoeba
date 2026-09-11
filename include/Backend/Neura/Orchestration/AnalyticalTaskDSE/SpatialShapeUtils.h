// Spatial candidate facts shared by analytical task DSE passes.

#ifndef AMOEBA_ANALYTICAL_DSE_SPATIAL_SHAPE_UTILS_H
#define AMOEBA_ANALYTICAL_DSE_SPATIAL_SHAPE_UTILS_H

#include "Backend/Neura/Orchestration/orchestration_utils.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Support/LogicalResult.h"

#include <cstdint>
#include <optional>
#include <string>

namespace mlir {
namespace taskflow {
namespace static_shape {

using ::mlir::taskflow::CgraShape;

// Generates every rectangular shape for cgra_count within the given grid.
// Shapes are ordered by ascending row count and then column count.
llvm::SmallVector<CgraShape>
getRectangularShapes(int cgra_count, int grid_rows = kCgraGridRows,
                     int grid_cols = kCgraGridCols);

// Infers a trip count from Taskflow counter chains whose bounds and steps are
// constant index values. Counts multiply along each root-to-leaf chain;
// concurrent sibling chains and independent roots use the maximum. Returns
// success(number) for supported static chains, success(std::nullopt) when the
// task has no Taskflow counter, and failure for non-constant, malformed, or
// overflowing counters.
FailureOr<std::optional<int64_t>> inferStaticTaskTripCount(TaskflowTaskOp task,
                                                           std::string &error);

} // namespace static_shape
} // namespace taskflow
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_DSE_SPATIAL_SHAPE_UTILS_H

//===- AnalyticalTaskCandidateSupport.h ----------------------*- C++ -*-===//
//
// Data records plus hashing and file/JSON support shared by analytical
// candidate-space implementations.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_BACKEND_NEURA_TRANSFORMS_OPTIMIZATIONS_ANALYTICAL_TASK_CANDIDATE_SUPPORT_H
#define AMOEBA_BACKEND_NEURA_TRANSFORMS_OPTIMIZATIONS_ANALYTICAL_TASK_CANDIDATE_SUPPORT_H

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Support/LLVM.h"

#include "llvm/ADT/FunctionExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"

#include <cstdint>
#include <string>

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_candidates {

inline constexpr llvm::StringLiteral kCandidateSchema =
    "amoeba-analytical-task-candidates";
inline constexpr llvm::StringLiteral kSourceTaskBodyShaAttr =
    "amoeba.source_task_body_sha256";

// Stores immutable identity and the compile-time trip count from one Taskflow
// task. Tasks without a Taskflow counter represent one execution.
struct TaskMetadata {
  taskflow::TaskflowTaskOp op;
  std::string name;
  std::string body_sha256;
  int64_t trip_count = 1;
};

FailureOr<std::string> currentArchitectureSha256(std::string &error);

std::string taskBodySha256(taskflow::TaskflowTaskOp task);

std::string makeSequentialCandidateId(uint64_t index);

// Writes one JSON object as one JSONL record.
void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object);

// Publishes a complete output atomically so consumers never read a partial
// candidate manifest.
bool writeAtomically(llvm::StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> write_body,
                     std::string &error);

} // namespace analytical_candidates
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_BACKEND_NEURA_TRANSFORMS_OPTIMIZATIONS_ANALYTICAL_TASK_CANDIDATE_SUPPORT_H

//===- AnalyticalTaskCandidateCommon.h -----------------------*- C++ -*-===//
//
// Shared facts and file/JSON helpers for analytical task candidate spaces.
// Candidate-space implementations provide their own shape and traversal
// types, so a future temporal space can reuse these utilities independently.
//
//===----------------------------------------------------------------------===//

#ifndef AMOEBA_ANALYTICAL_TASK_CANDIDATE_COMMON_H
#define AMOEBA_ANALYTICAL_TASK_CANDIDATE_COMMON_H

#include "TaskflowDialect/TaskflowOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
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
namespace analytical_dse {

inline constexpr llvm::StringLiteral kCandidateSchema =
    "amoeba-analytical-task-candidates";
inline constexpr llvm::StringLiteral kSourceTaskBodyShaAttr =
    "amoeba.source_task_body_sha256";

// Stores immutable identity and the compile-time trip count from one Taskflow
// task. Tasks without a Taskflow counter represent one execution.
struct TaskFact {
  taskflow::TaskflowTaskOp op;
  std::string name;
  std::string bodySha256;
  int64_t tripCount = 1;
};

FailureOr<llvm::SmallVector<TaskFact>>
collectAnalyticalTaskFacts(func::FuncOp func, std::string &error);

FailureOr<std::string> currentArchitectureSha256(std::string &error);

FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module,
                                           llvm::StringRef requested,
                                           std::string &error);

std::string makeSequentialCandidateId(uint64_t index);

// Writes one JSON object as one JSONL record.
void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object);

// Publishes a complete output atomically so consumers never read a partial
// candidate manifest.
bool writeAtomically(llvm::StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error);

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

#endif // AMOEBA_ANALYTICAL_TASK_CANDIDATE_COMMON_H

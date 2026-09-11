//===- AnalyticalTaskCandidateCommon.cpp ---------------------*- C++ -*-===//
//
// Implements reusable task facts, architecture fingerprints, and output
// helpers shared by analytical candidate-space implementations.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"

#include "Backend/Neura/NeuraBackendOptions.h"
#include "Backend/Neura/Orchestration/AnalyticalTaskDSE/SpatialDSEOrchestration.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

#include <functional>
#include <memory>
#include <optional>
#include <system_error>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

// Returns a lowercase SHA-256. File hashes use raw bytes so any input change
// invalidates a manifest produced from those bytes.
static std::string sha256(StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

// Fingerprints the exact YAML selected by --architecture-spec. Dimensions
// alone are insufficient because two same-sized machines can have different
// FU, memory, latency, or routing capabilities.
FailureOr<std::string> currentArchitectureSha256(std::string &error) {
  StringRef path = mlir::amoeba::getNeuraArchitectureSpecFile();
  if (path.empty()) {
    error = "analytical task DSE requires --architecture-spec";
    return failure();
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot fingerprint architecture specification " + path.str() +
            ": " + buffer.getError().message();
    return failure();
  }
  return sha256((*buffer)->getBuffer());
}

// Resolves any compile-time trip count stored with each task. An explicit
// `trip_count` is authoritative; otherwise, constant Taskflow counter chains
// supply the count. Dynamic or invalid bounds fail instead of inventing a
// numeric value. A task without a counter represents one execution.
static FailureOr<int64_t> resolveAnalyticalTripCount(TaskflowTaskOp task,
                                                     std::string &error) {
  if (auto attr = task->getAttrOfType<IntegerAttr>("trip_count")) {
    if (attr.getInt() <= 0) {
      error =
          "task " + task.getTaskName().str() + " has non-positive trip_count";
      return failure();
    }
    return attr.getInt();
  }

  FailureOr<std::optional<int64_t>> inferred =
      SpatialDSEOrchestration::inferStaticTaskTripCount(task, error);
  if (failed(inferred)) {
    error += "; add an explicit positive trip_count or resolve the counter "
             "bounds first";
    return failure();
  }
  return inferred->value_or(1);
}

// Produces a stable identity for the current task computation. We deliberately
// remove DSE outputs and measurements so materializing a shape does not make
// an otherwise identical task look new. A body edit, however, changes this
// hash and invalidates an old candidate manifest.
static std::string taskBodySha256(TaskflowTaskOp task) {
  Operation *clone = task->clone();
  auto destroyClone = llvm::make_scope_exit([&] { clone->destroy(); });
  clone->setAttr("task_name",
                 StringAttr::get(task.getContext(), "__analytical_task__"));
  for (StringRef attribute :
       {"trip_count", "cgra_count", "cgra_shape", "compiled_ii", "profile_info",
        "task_orchestration_info", "replicas", "tiling", "est_latency"})
    clone->removeAttr(attribute);
  clone->removeAttr("amoeba.analytical_shape_orientation_fixed");
  clone->removeAttr(kSourceTaskBodyShaAttr);
  std::string printed;
  llvm::raw_string_ostream stream(printed);
  OpPrintingFlags flags;
  flags.printGenericOpForm().useLocalScope();
  clone->print(stream, flags);
  stream.flush();
  return sha256(printed);
}

// Collects task names, operations, and available trip counts in walk order.
// The order is the task axis used by a spatial shape tuple, so duplicate names
// are rejected before they can make candidate records ambiguous.
FailureOr<SmallVector<TaskFact>>
collectAnalyticalTaskFacts(func::FuncOp func, std::string &error) {
  SmallVector<TaskFact> tasks;
  llvm::StringSet<> names;
  WalkResult walkResult = func.walk([&](TaskflowTaskOp task) {
    std::string name = task.getTaskName().str();
    if (!names.insert(name).second) {
      error = "duplicate task name " + name;
      return WalkResult::interrupt();
    }
    FailureOr<int64_t> tripCount = resolveAnalyticalTripCount(task, error);
    if (failed(tripCount))
      return WalkResult::interrupt();
    int64_t materializedOperationCount = 0;
    task.walk([&](Operation *operation) {
      StringRef opName = operation->getName().getStringRef();
      if (!opName.starts_with("neura.") || opName == "neura.kernel" ||
          opName == "neura.reserve" || opName == "neura.ctrl_mov" ||
          opName == "neura.data_mov" || opName == "neura.yield" ||
          opName == "neura.br" || opName == "neura.cond_br")
        return;
      ++materializedOperationCount;
    });
    tasks.push_back({task, std::move(name), taskBodySha256(task), *tripCount,
                     materializedOperationCount});
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();
  if (tasks.empty()) {
    error = "function contains no taskflow.task operations";
    return failure();
  }
  return tasks;
}

std::string makeSequentialCandidateId(uint64_t index) {
  return "candidate-" + std::to_string(index);
}

void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object) {
  os << llvm::json::Value(std::move(object)) << "\n";
}

// Publishes a complete output atomically so consumers never read a partial
// candidate manifest.
bool writeAtomically(StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> writeBody,
                     std::string &error) {
  if (output.empty()) {
    error = "output path is required";
    return false;
  }
  SmallString<256> pattern(output);
  pattern += ".tmp-%%%%%%";
  SmallString<256> temporary;
  int descriptor = -1;
  std::error_code ec =
      llvm::sys::fs::createUniqueFile(pattern, descriptor, temporary);
  if (ec) {
    error = "cannot create temporary output " + temporary.str().str() + ": " +
            ec.message();
    return false;
  }

  bool ok = false;
  {
    llvm::raw_fd_ostream os(descriptor, /*shouldClose=*/true);
    ok = writeBody(os);
    os.flush();
    if (os.has_error()) {
      error = "failed while writing " + output.str();
      ok = false;
    }
  }
  if (!ok) {
    llvm::sys::fs::remove(temporary);
    return false;
  }
  ec = llvm::sys::fs::rename(temporary, output);
  if (ec) {
    error = "cannot publish " + output.str() + ": " + ec.message();
    llvm::sys::fs::remove(temporary);
    return false;
  }
  return true;
}

// Selects the requested Taskflow function, or infers it when exactly one
// function contains tasks. This keeps every file-producing pass consistent.
FailureOr<func::FuncOp> selectTaskFunction(ModuleOp module, StringRef requested,
                                           std::string &error) {
  if (!requested.empty()) {
    auto function = module.lookupSymbol<func::FuncOp>(requested);
    if (!function) {
      error = "requested function " + requested.str() + " does not exist";
      return failure();
    }
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (!hasTask) {
      error = "requested function " + requested.str() +
              " contains no taskflow.task operations";
      return failure();
    }
    return function;
  }

  SmallVector<func::FuncOp> taskFunctions;
  for (func::FuncOp function : module.getOps<func::FuncOp>()) {
    bool hasTask = false;
    function.walk([&](TaskflowTaskOp) { hasTask = true; });
    if (hasTask)
      taskFunctions.push_back(function);
  }
  if (taskFunctions.size() != 1) {
    error = "expected exactly one function containing Taskflow tasks; use the "
            "function option when the module contains more than one";
    return failure();
  }
  return taskFunctions.front();
}

} // namespace analytical_dse
} // namespace neura
} // namespace amoeba
} // namespace mlir

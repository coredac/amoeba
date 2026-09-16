//===- AnalyticalTaskCandidateCommon.cpp ---------------------*- C++ -*-===//
//
// Implements reusable task metadata, architecture fingerprints, and output
// helpers shared by analytical candidate-space implementations.
//
//===----------------------------------------------------------------------===//

#include "AnalyticalTaskCandidateCommon.h"

#include "Backend/Neura/NeuraBackendOptions.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/SymbolTable.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <system_error>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_dse {

// Infers the execution count recorded in analytical candidate manifests from
// a task's static Taskflow counter forest. Each counter iterates over the
// half-open range [lower, upper) with a positive step. Nested counters multiply
// their counts; sibling chains and independent roots contribute their maximum
// chain count. Spatial and future temporal candidate spaces use the same task
// metadata, so counter interpretation belongs to this shared candidate layer.
// The routine fails for dynamic bounds, empty ranges, malformed counter graphs,
// or int64_t overflow.
static FailureOr<std::optional<int64_t>>
inferStaticTaskTripCount(TaskflowTaskOp task, std::string &error) {
  // Collects every counter owned by the task. No counters means the manifest
  // may use the caller's default of one execution.
  SmallVector<TaskflowCounterOp> counters;
  task.walk([&](TaskflowCounterOp counter) { counters.push_back(counter); });
  if (counters.empty()) {
    return std::optional<int64_t>{};
  }

  if (!task.getBody().hasOneBlock()) {
    error = "task " + task.getTaskName().str() +
            " must contain exactly one block to infer a static trip count";
    return failure();
  }

  // Reconstructs the counter forest from parent-index SSA values. Roots have
  // no parent index; children are keyed by the counter index they reference.
  SmallVector<TaskflowCounterOp> roots;
  llvm::DenseMap<Value, SmallVector<TaskflowCounterOp>> children;
  for (TaskflowCounterOp counter : counters) {
    if (Value parent = counter.getParentIndex()) {
      children[parent].push_back(counter);
    } else {
      roots.push_back(counter);
    }
  }
  if (roots.empty()) {
    error = "task " + task.getTaskName().str() +
            " has counters but no root counter";
    return failure();
  }

  // Accepts only arith.constant index bounds so the inferred count is stable
  // before candidate enumeration and independent of runtime values.
  auto constant_index = [](Value value) -> FailureOr<int64_t> {
    if (auto constant = value.getDefiningOp<arith::ConstantIndexOp>()) {
      return constant.value();
    }
    return failure();
  };
  // Computes ceil((upper - lower) / step) for one non-empty counter range.
  // The lower-bound check keeps upper - lower representable in int64_t.
  auto counter_trip_count =
      [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    FailureOr<int64_t> lower = constant_index(counter.getLowerBound());
    FailureOr<int64_t> upper = constant_index(counter.getUpperBound());
    FailureOr<int64_t> step = constant_index(counter.getStep());
    if (failed(lower) || failed(upper) || failed(step)) {
      return failure();
    }
    if (*step <= 0 || *upper <= *lower ||
        (*lower < 0 && *upper > std::numeric_limits<int64_t>::max() + *lower)) {
      return failure();
    }

    int64_t distance = *upper - *lower;
    return 1 + (distance - 1) / *step;
  };

  // Evaluates each nested chain with DFS. active detects a cycle in the
  // current path, while visited rejects a counter reached through multiple
  // parent paths. A parent's count multiplies the longest nested child chain.
  llvm::DenseSet<Operation *> active;
  llvm::DenseSet<Operation *> visited;
  std::function<FailureOr<int64_t>(TaskflowCounterOp)> chain_trip_count =
      [&](TaskflowCounterOp counter) -> FailureOr<int64_t> {
    Operation *operation = counter.getOperation();
    if (!active.insert(operation).second || visited.contains(operation)) {
      error = "task " + task.getTaskName().str() +
              " has a cyclic or multiply referenced counter chain";
      return failure();
    }

    FailureOr<int64_t> count = counter_trip_count(counter);
    if (failed(count)) {
      error = "task " + task.getTaskName().str() +
              " requires constant counter bounds, a positive step, a "
              "non-empty range, and a trip count within int64";
      return failure();
    }

    int64_t longest_child_chain = 1;
    auto found = children.find(counter.getCounterIndex());
    if (found != children.end()) {
      for (TaskflowCounterOp child : found->second) {
        FailureOr<int64_t> child_count = chain_trip_count(child);
        if (failed(child_count)) {
          return failure();
        }
        longest_child_chain = std::max(longest_child_chain, *child_count);
      }
    }
    if (*count > std::numeric_limits<int64_t>::max() / longest_child_chain) {
      error = "task " + task.getTaskName().str() +
              " requires constant counter bounds, a positive step, a "
              "non-empty range, and a trip count within int64";
      return failure();
    }

    active.erase(operation);
    visited.insert(operation);
    return *count * longest_child_chain;
  };

  // Multiple roots represent independent counter chains, so the task count is
  // the maximum root-chain count rather than their product.
  int64_t total = 1;
  for (TaskflowCounterOp root : roots) {
    FailureOr<int64_t> root_count = chain_trip_count(root);
    if (failed(root_count)) {
      return failure();
    }
    total = std::max(total, *root_count);
  }
  // Every collected counter must be reachable from exactly one root.
  if (visited.size() != counters.size()) {
    error = "task " + task.getTaskName().str() +
            " has a counter disconnected from every root";
    return failure();
  }
  return std::optional<int64_t>{total};
}

// Returns a lowercase SHA-256. File hashes use raw bytes so any input change
// invalidates a manifest produced from those bytes.
static std::string sha256(StringRef bytes) {
  llvm::SHA256 hasher;
  hasher.update(bytes);
  return llvm::toHex(hasher.final(), /*LowerCase=*/true);
}

// Hashes the exact YAML selected by --architecture-spec. The architecture
// hash is stored in the candidate manifest and checked by every downstream
// consumer: same-sized machines can still differ in functional units,
// memory, latency, or routing, so any YAML change invalidates candidates and
// their derived predictions.
FailureOr<std::string> currentArchitectureSha256(std::string &error) {
  StringRef path = mlir::amoeba::getNeuraArchitectureSpecFile();
  if (path.empty()) {
    error = "analytical task DSE requires --architecture-spec";
    return failure();
  }
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> buffer =
      llvm::MemoryBuffer::getFile(path);
  if (!buffer) {
    error = "cannot hash architecture specification " + path.str() + ": " +
            buffer.getError().message();
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
      inferStaticTaskTripCount(task, error);
  if (failed(inferred)) {
    error += "; add an explicit positive trip_count or resolve the counter "
             "bounds first";
    return failure();
  }
  return inferred->value_or(1);
}

// Produces the task identity consumed by candidate and prediction manifests.
// The enumerator writes this hash into each task record and attaches it to the
// bound IR; materialization and the predictor use it to bind derived data to
// the source task. A task-body edit changes the hash, so manifest validation
// rejects old candidates and predictions. DSE outputs and measurements are
// deliberately excluded so materializing or measuring a shape does not make
// the same source computation look new.
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

// Collects task names, source-body identities, and available trip counts in
// walk order.
// The order is the task axis used by a spatial shape tuple, so duplicate names
// are rejected before they can make candidate records ambiguous.
FailureOr<SmallVector<TaskMetadata>>
collectAnalyticalTaskMetadata(func::FuncOp func, std::string &error) {
  SmallVector<TaskMetadata> tasks;
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
    tasks.push_back({task, std::move(name), taskBodySha256(task), *tripCount});
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

//===- AnalyticalTaskCandidateSupport.cpp --------------------*- C++ -*-===//
//
// Implements reusable architecture/task fingerprints and file/JSON support
// shared by analytical candidate-space implementations.
//
//===----------------------------------------------------------------------===//

#include "Backend/Neura/Transforms/Optimizations/AnalyticalTaskCandidateSupport.h"

#include "Backend/Neura/NeuraBackendOptions.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SHA256.h"

#include <memory>
#include <system_error>

using namespace mlir;
using namespace mlir::taskflow;

namespace mlir {
namespace amoeba {
namespace neura {
namespace analytical_candidates {

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
    error = "analytical task candidate enumeration requires "
            "--architecture-spec";
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

// Produces the task identity consumed by candidate and prediction manifests.
// The enumerator writes this hash into each task record and attaches it to the
// bound IR; materialization and the predictor use it to bind derived data to
// the source task. A task-body edit changes the hash, so manifest validation
// rejects old candidates and predictions. Candidate outputs and measurements
// are deliberately excluded so materializing or measuring a shape does not
// make the same source computation look new.
std::string taskBodySha256(TaskflowTaskOp task) {
  Operation *clone = task->clone();
  auto destroy_clone = llvm::make_scope_exit([&] { clone->destroy(); });
  clone->setAttr("task_name",
                 StringAttr::get(task.getContext(), "__analytical_task__"));
  for (StringRef attribute :
       {"trip_count", "cgra_count", "cgra_shape", "compiled_ii", "profile_info",
        "task_orchestration_info", "replicas", "tiling", "est_latency"}) {
    clone->removeAttr(attribute);
  }
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

std::string makeSequentialCandidateId(uint64_t index) {
  return "candidate-" + std::to_string(index);
}

void writeJsonLine(llvm::raw_ostream &os, llvm::json::Object object) {
  os << llvm::json::Value(std::move(object)) << "\n";
}

// Publishes a complete output atomically so consumers never read a partial
// candidate manifest.
bool writeAtomically(StringRef output,
                     llvm::function_ref<bool(llvm::raw_ostream &)> write_body,
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
    ok = write_body(os);
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

} // namespace analytical_candidates
} // namespace neura
} // namespace amoeba
} // namespace mlir

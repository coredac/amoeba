// RUN: mlir-amoeba-opt %s --orchestrate-tasks-on-accelerators="orchestration-strategy=unknown" --verify-diagnostics
// RUN: mlir-amoeba-opt %s --orchestrate-tasks-on-accelerators="orchestration-strategy=throughput-guided" --verify-diagnostics

// Unknown or unavailable strategies must fail instead of silently falling back.
// expected-error@+1 {{unknown task orchestration strategy:}}
func.func @unknown_strategy() {
  return
}

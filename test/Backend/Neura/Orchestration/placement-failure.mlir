// RUN: mlir-amoeba-opt %s --orchestrate-tasks-on-accelerators="scheduling-mode=spatial" --neura-architecture-spec=%S/../../../archspec/architecture_1x2.yaml --verify-diagnostics

// The first task fills the grid, so the second cannot be placed spatially.
// Propagate the scheduler failure as a pass diagnostic, without an assertion.
// expected-error@+1 {{failed to orchestrate taskflow tasks with strategy: routing-critical-path}}
func.func @spatial_overflow() {
  taskflow.task @first {cgra_count = 2 : i32} : () -> () {
    taskflow.yield
  }
  taskflow.task @second {cgra_count = 1 : i32} : () -> () {
    taskflow.yield
  }
  return
}

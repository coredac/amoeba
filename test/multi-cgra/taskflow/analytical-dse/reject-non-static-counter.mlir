// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   --verify-diagnostics

module {
  // expected-error@+1 {{task A requires constant counter bounds, a positive step, a non-empty range, and a trip count within int64; static shape selection requires constant counter bounds}}
  func.func @main(%upper: index, %input: i32) -> i32 {
    %result = taskflow.task @A value_inputs(%upper, %input : index, i32)
        : (index, i32) -> i32 {
    ^bb0(%task_upper: index, %task_input: i32):
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i = taskflow.counter from %c0 to %task_upper step %c1 : index
      taskflow.yield values(%task_input : i32)
    }
    return %result : i32
  }
}

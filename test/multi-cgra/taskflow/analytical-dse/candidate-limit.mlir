// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl max-cgras-per-task=4 max-candidates=7' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   --verify-diagnostics

module {
  // expected-error@+1 {{complete concurrently packable shape space exceeds max-candidates=7; refusing to publish a partial candidate manifest}}
  func.func @main(%input: i32) -> i32 {
    %result = taskflow.task @A value_inputs(%input : i32)
        {trip_count = 10 : i64} : (i32) -> i32 {
    ^bb0(%task_input: i32):
      taskflow.yield values(%task_input : i32)
    }
    return %result : i32
  }
}

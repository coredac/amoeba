// XFAIL: *
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_1x2.yaml \
// RUN:   --mlir-print-op-on-diagnostic=false -o /dev/null
// RUN: sed '/"record_type":"candidate"/d' %t.candidates.jsonl \
// RUN:   > %t.incomplete.jsonl
// RUN: mlir-amoeba-opt %s \
// RUN:   '--materialize-analytical-task-candidate=candidates=%t.incomplete.jsonl candidate-id=candidate-0' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_1x2.yaml \
// RUN:   --mlir-print-op-on-diagnostic=false

module {
  func.func @main(%input: i32) -> i32 {
    %result = taskflow.task @A value_inputs(%input : i32)
        {trip_count = 1 : i64} : (i32) -> i32 {
    ^bb0(%task_input: i32):
      taskflow.yield values(%task_input : i32)
    }
    return %result : i32
  }
}

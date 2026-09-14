// Two dependent tasks each take five predicted cycles. A compute-bottleneck
// score would be five; the production spatial-temporal scheduler makespan is ten.
// RUN: mlir-amoeba-opt %s \
// RUN:   '--enumerate-analytical-task-candidates=output=%t.candidates.jsonl' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null
// RUN: python3 %S/fixed-makespan-catalog.py create \
// RUN:   %t.candidates.jsonl %t.cost.json
// RUN: mlir-amoeba-opt %s \
// RUN:   '--score-analytical-task-candidates=candidates=%t.candidates.jsonl cost-file=%t.cost.json output=%t.scores.jsonl top-k=1' \
// RUN:   --architecture-spec=%S/../../../archspec/architecture_4x4.yaml \
// RUN:   -o /dev/null
// RUN: python3 %S/fixed-makespan-catalog.py verify \
// RUN:   %t.candidates.jsonl %t.cost.json %t.scores.jsonl

module {
  func.func @main(%seed: i32) -> i32 {
    %a = taskflow.task @A value_inputs(%seed : i32)
        {trip_count = 1 : i64} : (i32) -> i32 {
      ^bb0(%value: i32):
        taskflow.yield values(%value : i32)
    }
    %b = taskflow.task @B value_inputs(%a : i32)
        {trip_count = 1 : i64} : (i32) -> i32 {
      ^bb0(%value: i32):
        taskflow.yield values(%value : i32)
    }
    return %b : i32
  }
}

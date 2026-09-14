// Pipeline interval analysis for scheduled Taskflow programs.

#ifndef AMOEBA_NEURA_PIPELINE_INTERVAL_H
#define AMOEBA_NEURA_PIPELINE_INTERVAL_H

#include "TaskflowDialect/TaskflowOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>

namespace mlir {
namespace taskflow {

// Concrete schedule facts for one Taskflow task.
struct TaskScheduleResult {
  struct CgraOccupancy {
    int row = 0;
    int col = 0;
    int64_t start_time = 0;
    int64_t duration = 1;
    int context_id = 0;
  };

  TaskflowTaskOp task;
  int64_t start_time = 0;
  int64_t duration = 1;
  int64_t end_time = 1;
  llvm::SmallVector<CgraOccupancy> cgra_occupancies;
  llvm::SmallVector<TaskflowTaskOp> predecessor_tasks;
  llvm::SmallVector<TaskflowTaskOp> successor_tasks;
};

// Result of analyzing the steady-state interval of one scheduled function.
struct TaskPipelineIntervalResult {
  int64_t pipeline_interval = 0;
  TaskflowTaskOp bottleneck_task;
  llvm::SmallVector<TaskflowTaskOp> critical_path;
};

// Computes the longest execution path between tasks that reuse a CGRA. The
// input schedule is the concrete result emitted by TaskScheduler, including
// task data-dependence edges and per-cell context order.
class TaskPipelineIntervalAnalyzer {
public:
  explicit TaskPipelineIntervalAnalyzer(
      llvm::ArrayRef<TaskScheduleResult> schedule_result);

  TaskPipelineIntervalResult analyze();

private:
  struct ExecutionOrderEdge {
    int next_task_idx = -1;
    int64_t latency = 0;
  };

  struct CgraPipelineCycle {
    int last_task_idx = -1;
    int first_task_idx = -1;
    int64_t latency = 0;
  };

  struct LongestExecutionPath {
    bool found = false;
    int64_t total_latency = 0;
    llvm::SmallVector<int> path;
  };

  int64_t getTaskDuration(int task_idx) const;
  void addExecutionOrderEdge(int task_idx, int next_task_idx);
  void buildTaskIndex();
  void buildDataDependenceEdges();
  void buildCgraExecutionOrderEdgesAndPipelineCycles();
  LongestExecutionPath findLongestPathToTarget(
      int current_task_idx, int target_task_idx, llvm::DenseSet<int> &visiting,
      llvm::DenseMap<int, LongestExecutionPath> &memo) const;
  TaskPipelineIntervalResult computeLongestPipelineCycle() const;

  llvm::ArrayRef<TaskScheduleResult> schedule_result_;
  llvm::DenseMap<Operation *, int> task_to_index_;
  llvm::SmallVector<llvm::SmallVector<ExecutionOrderEdge>> task_graph_;
  llvm::SmallVector<CgraPipelineCycle> cgra_pipeline_cycles_;
};

} // namespace taskflow
} // namespace mlir

#endif // AMOEBA_NEURA_PIPELINE_INTERVAL_H

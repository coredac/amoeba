#include "Backend/Neura/Orchestration/PipelineInterval/PipelineInterval.h"

#include "llvm/ADT/SmallVector.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

static int64_t encodePipelineCgraLocation(int row, int col) {
  return (static_cast<int64_t>(row) << 32) | static_cast<uint32_t>(col);
}

} // namespace

TaskPipelineIntervalAnalyzer::TaskPipelineIntervalAnalyzer(
    llvm::ArrayRef<TaskScheduleResult> schedule_result)
    : schedule_result_(schedule_result) {}

TaskPipelineIntervalResult TaskPipelineIntervalAnalyzer::analyze() {
  TaskPipelineIntervalResult result;
  if (schedule_result_.empty()) {
    return result;
  }

  buildTaskIndex();
  task_graph_.resize(schedule_result_.size());
  buildDataDependenceEdges();
  buildCgraExecutionOrderEdgesAndPipelineCycles();
  return computeLongestPipelineCycle();
}

int64_t TaskPipelineIntervalAnalyzer::getTaskDuration(int task_idx) const {
  return std::max<int64_t>(1, schedule_result_[task_idx].duration);
}

void TaskPipelineIntervalAnalyzer::addExecutionOrderEdge(int task_idx,
                                                         int next_task_idx) {
  if (task_idx < 0 || next_task_idx < 0 || task_idx == next_task_idx) {
    return;
  }
  task_graph_[task_idx].push_back({next_task_idx, getTaskDuration(task_idx)});
}

void TaskPipelineIntervalAnalyzer::buildTaskIndex() {
  for (auto [idx, task_result] : llvm::enumerate(schedule_result_)) {
    TaskflowTaskOp task = task_result.task;
    task_to_index_[task.getOperation()] = static_cast<int>(idx);
  }
}

void TaskPipelineIntervalAnalyzer::buildDataDependenceEdges() {
  for (auto [task_idx, task_result] : llvm::enumerate(schedule_result_)) {
    for (TaskflowTaskOp pred : task_result.predecessor_tasks) {
      auto pred_it = task_to_index_.find(pred.getOperation());
      if (pred_it == task_to_index_.end()) {
        continue;
      }
      addExecutionOrderEdge(pred_it->second, static_cast<int>(task_idx));
    }
  }
}

void TaskPipelineIntervalAnalyzer::
    buildCgraExecutionOrderEdgesAndPipelineCycles() {
  llvm::DenseMap<int64_t, llvm::SmallVector<int>> cgra_location_to_tasks;
  for (auto [idx, task_result] : llvm::enumerate(schedule_result_)) {
    for (const TaskScheduleResult::CgraOccupancy &occupancy :
         task_result.cgra_occupancies) {
      cgra_location_to_tasks[encodePipelineCgraLocation(occupancy.row,
                                                        occupancy.col)]
          .push_back(static_cast<int>(idx));
    }
  }

  // DenseMap iteration order is intentionally unspecified. Sort the physical
  // locations before selecting a tied critical path so the emitted result is
  // stable across runs and library builds.
  llvm::SmallVector<int64_t> locations;
  locations.reserve(cgra_location_to_tasks.size());
  for (const auto &entry : cgra_location_to_tasks) {
    locations.push_back(entry.first);
  }
  llvm::sort(locations);

  for (int64_t location : locations) {
    llvm::SmallVector<int> &tasks = cgra_location_to_tasks[location];
    llvm::sort(tasks, [&](int lhs, int rhs) {
      const TaskScheduleResult &lhs_result = schedule_result_[lhs];
      const TaskScheduleResult &rhs_result = schedule_result_[rhs];
      if (lhs_result.start_time != rhs_result.start_time) {
        return lhs_result.start_time < rhs_result.start_time;
      }
      return lhs < rhs;
    });

    for (size_t i = 1; i < tasks.size(); ++i) {
      addExecutionOrderEdge(tasks[i - 1], tasks[i]);
    }

    if (!tasks.empty()) {
      int first_task_idx = tasks.front();
      int last_task_idx = tasks.back();
      cgra_pipeline_cycles_.push_back(
          {last_task_idx, first_task_idx, getTaskDuration(last_task_idx)});
    }
  }
}

TaskPipelineIntervalAnalyzer::LongestExecutionPath
TaskPipelineIntervalAnalyzer::findLongestPathToTarget(
    int current_task_idx, int target_task_idx, llvm::DenseSet<int> &visiting,
    llvm::DenseMap<int, LongestExecutionPath> &memo) const {
  if (current_task_idx == target_task_idx) {
    LongestExecutionPath result;
    result.found = true;
    result.path.push_back(current_task_idx);
    return result;
  }

  if (visiting.contains(current_task_idx)) {
    return LongestExecutionPath();
  }

  auto memo_it = memo.find(current_task_idx);
  if (memo_it != memo.end()) {
    return memo_it->second;
  }

  visiting.insert(current_task_idx);
  LongestExecutionPath best;
  for (const ExecutionOrderEdge &edge : task_graph_[current_task_idx]) {
    LongestExecutionPath suffix = findLongestPathToTarget(
        edge.next_task_idx, target_task_idx, visiting, memo);
    if (!suffix.found) {
      continue;
    }

    int64_t total_latency = edge.latency + suffix.total_latency;
    if (!best.found || total_latency > best.total_latency) {
      best.found = true;
      best.total_latency = total_latency;
      best.path.clear();
      best.path.push_back(current_task_idx);
      best.path.append(suffix.path.begin(), suffix.path.end());
    }
  }
  visiting.erase(current_task_idx);
  memo[current_task_idx] = best;
  return best;
}

TaskPipelineIntervalResult
TaskPipelineIntervalAnalyzer::computeLongestPipelineCycle() const {
  TaskPipelineIntervalResult result;
  for (const CgraPipelineCycle &pipeline_cycle : cgra_pipeline_cycles_) {
    llvm::DenseSet<int> visiting;
    llvm::DenseMap<int, LongestExecutionPath> memo;
    LongestExecutionPath path =
        findLongestPathToTarget(pipeline_cycle.first_task_idx,
                                pipeline_cycle.last_task_idx, visiting, memo);
    if (!path.found) {
      continue;
    }

    int64_t interval = path.total_latency + pipeline_cycle.latency;
    if (interval <= result.pipeline_interval) {
      continue;
    }

    result.pipeline_interval = interval;
    result.critical_path.clear();

    int bottleneck_idx = pipeline_cycle.last_task_idx;
    int64_t bottleneck_duration = getTaskDuration(bottleneck_idx);
    for (int idx : path.path) {
      const TaskScheduleResult &task_result = schedule_result_[idx];
      result.critical_path.push_back(task_result.task);
      int64_t duration = getTaskDuration(idx);
      if (duration > bottleneck_duration) {
        bottleneck_idx = idx;
        bottleneck_duration = duration;
      }
    }
    result.bottleneck_task = schedule_result_[bottleneck_idx].task;
  }
  return result;
}

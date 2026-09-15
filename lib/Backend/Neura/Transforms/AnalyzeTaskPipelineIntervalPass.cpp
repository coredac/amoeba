// Analyze the steady-state interval of a scheduled Taskflow pipeline.

#include "Backend/Neura/NeuraBackendPasses.h"
#include "Backend/Neura/Orchestration/PipelineInterval/PipelineInterval.h"
#include "TaskflowDialect/TaskflowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::taskflow;

namespace {

static int64_t encodePipelineCgraLocation(int row, int col) {
  return (static_cast<int64_t>(row) << 32) | static_cast<uint32_t>(col);
}

static llvm::SmallVector<TaskflowTaskOp>
collectPipelineTasks(func::FuncOp func) {
  llvm::SmallVector<TaskflowTaskOp> tasks;
  func.walk([&](TaskflowTaskOp task) { tasks.push_back(task); });
  return tasks;
}

static int getPipelineTaskTileGroup(TaskflowTaskOp task) {
  if (auto group = task->getAttrOfType<IntegerAttr>("tile_group")) {
    return static_cast<int>(group.getInt());
  }
  return -1;
}

static bool isPipelineParallelTaskTile(TaskflowTaskOp task) {
  auto parallel = task->getAttrOfType<BoolAttr>("tile_parallel");
  return !parallel || parallel.getValue();
}

static int64_t getRequiredTaskDuration(TaskflowTaskOp task) {
  if (auto est_latency_attr = task->getAttrOfType<IntegerAttr>("est_latency")) {
    int64_t est_latency_cycles = est_latency_attr.getInt();
    if (est_latency_cycles > 0) {
      return est_latency_cycles;
    }
  }

  auto profile_info = task->getAttrOfType<DictionaryAttr>("profile_info");
  if (!profile_info) {
    task.emitOpError() << "requires profile_info.duration";
    return -1;
  }

  auto duration = dyn_cast_or_null<IntegerAttr>(profile_info.get("duration"));
  if (!duration) {
    task.emitOpError() << "requires profile_info.duration";
    return -1;
  }

  auto ii_attr = task->getAttrOfType<IntegerAttr>("compiled_ii");
  auto trip_attr = task->getAttrOfType<IntegerAttr>("trip_count");
  if (ii_attr && trip_attr) {
    const int64_t ii = ii_attr.getInt();
    const int64_t trip = trip_attr.getInt();
    const int64_t steps = duration.getInt();
    if (ii > 0 && trip > 0 && steps > 0) {
      return std::max<int64_t>(1, ii * (trip - 1) + steps);
    }
  }

  return std::max<int64_t>(1, duration.getInt());
}

static std::optional<TaskScheduleResult::CgraOccupancy>
parseCgraOccupancy(TaskflowTaskOp task, Attribute attr) {
  auto coord = dyn_cast<DictionaryAttr>(attr);
  if (!coord) {
    task.emitOpError() << "requires dictionary entries in "
                          "task_orchestration_info.cgra_positions";
    return std::nullopt;
  }

  auto row = dyn_cast_or_null<IntegerAttr>(coord.get("row"));
  auto col = dyn_cast_or_null<IntegerAttr>(coord.get("col"));
  auto context_id = dyn_cast_or_null<IntegerAttr>(coord.get("context_id"));
  if (!row || !col || !context_id) {
    task.emitOpError() << "requires row, col, and context_id in each "
                          "task_orchestration_info.cgra_positions entry";
    return std::nullopt;
  }

  TaskScheduleResult::CgraOccupancy occupancy;
  occupancy.row = static_cast<int>(row.getInt());
  occupancy.col = static_cast<int>(col.getInt());
  occupancy.context_id = static_cast<int>(context_id.getInt());
  return occupancy;
}

static LogicalResult readTaskCgraOccupancies(
    TaskflowTaskOp task,
    llvm::SmallVectorImpl<TaskScheduleResult::CgraOccupancy> &occupancies) {
  auto orchestration_info =
      task->getAttrOfType<DictionaryAttr>("task_orchestration_info");
  if (!orchestration_info) {
    return task.emitOpError() << "requires task_orchestration_info";
  }

  auto cgra_positions =
      dyn_cast_or_null<ArrayAttr>(orchestration_info.get("cgra_positions"));
  if (!cgra_positions || cgra_positions.empty()) {
    return task.emitOpError()
           << "requires non-empty task_orchestration_info.cgra_positions";
  }

  for (Attribute attr : cgra_positions) {
    std::optional<TaskScheduleResult::CgraOccupancy> occupancy =
        parseCgraOccupancy(task, attr);
    if (!occupancy) {
      return failure();
    }
    occupancies.push_back(*occupancy);
  }
  return success();
}

class TaskScheduleResultBuilder {
public:
  explicit TaskScheduleResultBuilder(func::FuncOp func) : func_(func) {}

  FailureOr<llvm::SmallVector<TaskScheduleResult>> build() {
    collectTasks();
    if (failed(buildScheduleResults()) || failed(buildDependences()) ||
        failed(buildCgraExecutionOrderDependences()) ||
        failed(computeStartTimes())) {
      return failure();
    }
    updateScheduleTimes();
    return std::move(schedule_result_);
  }

private:
  void collectTasks() {
    for (TaskflowTaskOp task : collectPipelineTasks(func_)) {
      task_to_index_[task.getOperation()] = tasks_.size();
      tasks_.push_back(task);
    }
  }

  LogicalResult buildScheduleResults() {
    for (TaskflowTaskOp task : tasks_) {
      int64_t duration = getRequiredTaskDuration(task);
      if (duration < 0) {
        return failure();
      }

      TaskScheduleResult task_result;
      task_result.task = task;
      task_result.duration = duration;
      task_result.end_time = duration;
      if (failed(readTaskCgraOccupancies(task, task_result.cgra_occupancies))) {
        return failure();
      }
      for (TaskScheduleResult::CgraOccupancy &occupancy :
           task_result.cgra_occupancies) {
        occupancy.duration = duration;
      }
      schedule_result_.push_back(std::move(task_result));
    }
    return success();
  }

  void addExecutionOrderDependence(int task, int next_task) {
    if (task < 0 || next_task < 0 || task == next_task) {
      return;
    }

    int64_t edge_key =
        (static_cast<int64_t>(task) << 32) | static_cast<uint32_t>(next_task);
    if (!execution_order_edge_keys_.insert(edge_key).second) {
      return;
    }

    successors_[task].push_back(next_task);
    ++predecessor_count_[next_task];
  }

  static void appendUnique(llvm::SmallVectorImpl<TaskflowTaskOp> &tasks,
                           TaskflowTaskOp task) {
    if (!llvm::is_contained(tasks, task)) {
      tasks.push_back(task);
    }
  }

  LogicalResult buildDependences() {
    successors_.resize(tasks_.size());
    predecessor_count_.assign(tasks_.size(), 0);
    start_times_.assign(tasks_.size(), 0);

    for (auto [task_idx, task] : llvm::enumerate(tasks_)) {
      const int task_tile_group = getPipelineTaskTileGroup(task);
      for (Value operand : task->getOperands()) {
        auto producer = operand.getDefiningOp<TaskflowTaskOp>();
        if (!producer) {
          continue;
        }

        if (task_tile_group >= 0 &&
            getPipelineTaskTileGroup(producer) == task_tile_group &&
            isPipelineParallelTaskTile(task)) {
          continue;
        }

        const int producer_tile_group = getPipelineTaskTileGroup(producer);
        if (producer_tile_group >= 0 &&
            producer_tile_group != task_tile_group) {
          for (auto [sibling_idx, sibling_task] : llvm::enumerate(tasks_)) {
            if (sibling_task == producer ||
                getPipelineTaskTileGroup(sibling_task) != producer_tile_group) {
              continue;
            }
            addExecutionOrderDependence(static_cast<int>(sibling_idx),
                                        static_cast<int>(task_idx));
            appendUnique(schedule_result_[task_idx].predecessor_tasks,
                         sibling_task);
            appendUnique(schedule_result_[sibling_idx].successor_tasks, task);
          }
        }

        auto producer_it = task_to_index_.find(producer.getOperation());
        if (producer_it == task_to_index_.end()) {
          continue;
        }

        const int producer_idx = producer_it->second;
        addExecutionOrderDependence(producer_idx, static_cast<int>(task_idx));
        appendUnique(schedule_result_[task_idx].predecessor_tasks, producer);
        appendUnique(schedule_result_[producer_idx].successor_tasks, task);
      }
    }
    return success();
  }

  int getTaskContextOnCgra(int task_idx, int64_t cgra_location) const {
    for (const TaskScheduleResult::CgraOccupancy &occupancy :
         schedule_result_[task_idx].cgra_occupancies) {
      if (encodePipelineCgraLocation(occupancy.row, occupancy.col) ==
          cgra_location) {
        return occupancy.context_id;
      }
    }
    return 0;
  }

  LogicalResult buildCgraExecutionOrderDependences() {
    llvm::DenseMap<int64_t, llvm::SmallVector<int>> cgra_location_to_tasks;
    for (auto [task_idx, task_result] : llvm::enumerate(schedule_result_)) {
      for (const TaskScheduleResult::CgraOccupancy &occupancy :
           task_result.cgra_occupancies) {
        cgra_location_to_tasks[encodePipelineCgraLocation(occupancy.row,
                                                          occupancy.col)]
            .push_back(static_cast<int>(task_idx));
      }
    }

    llvm::SmallVector<int64_t> locations;
    locations.reserve(cgra_location_to_tasks.size());
    for (const auto &entry : cgra_location_to_tasks) {
      locations.push_back(entry.first);
    }
    llvm::sort(locations);

    for (int64_t location : locations) {
      llvm::SmallVector<int> &tasks = cgra_location_to_tasks[location];
      llvm::sort(tasks, [&](int lhs, int rhs) {
        int lhs_context = getTaskContextOnCgra(lhs, location);
        int rhs_context = getTaskContextOnCgra(rhs, location);
        if (lhs_context != rhs_context) {
          return lhs_context < rhs_context;
        }
        return lhs < rhs;
      });

      for (size_t i = 1; i < tasks.size(); ++i) {
        addExecutionOrderDependence(tasks[i - 1], tasks[i]);
      }
    }
    return success();
  }

  LogicalResult computeStartTimes() {
    llvm::SmallVector<int> ready_tasks;
    for (auto [task_idx, pred_count] : llvm::enumerate(predecessor_count_)) {
      if (pred_count == 0) {
        ready_tasks.push_back(static_cast<int>(task_idx));
      }
    }

    size_t processed_tasks = 0;
    while (!ready_tasks.empty()) {
      int task = ready_tasks.pop_back_val();
      ++processed_tasks;

      for (int next_task : successors_[task]) {
        int64_t next_start =
            start_times_[task] + schedule_result_[task].duration;
        start_times_[next_task] = std::max(start_times_[next_task], next_start);

        --predecessor_count_[next_task];
        if (predecessor_count_[next_task] == 0) {
          ready_tasks.push_back(next_task);
        }
      }
    }

    if (processed_tasks != tasks_.size()) {
      return func_.emitError()
             << "cannot analyze task pipeline interval because task "
                "dependences and CGRA context order form a cycle";
    }
    return success();
  }

  void updateScheduleTimes() {
    for (auto [task_idx, task_result] : llvm::enumerate(schedule_result_)) {
      int64_t start_time = start_times_[task_idx];
      task_result.start_time = start_time;
      task_result.end_time = start_time + task_result.duration;
      for (TaskScheduleResult::CgraOccupancy &occupancy :
           task_result.cgra_occupancies) {
        occupancy.start_time = start_time;
        occupancy.duration = task_result.duration;
      }
    }
  }

  func::FuncOp func_;
  llvm::SmallVector<TaskflowTaskOp> tasks_;
  llvm::DenseMap<Operation *, int> task_to_index_;
  llvm::DenseSet<int64_t> execution_order_edge_keys_;
  llvm::SmallVector<llvm::SmallVector<int>> successors_;
  llvm::SmallVector<int> predecessor_count_;
  llvm::SmallVector<int64_t> start_times_;
  llvm::SmallVector<TaskScheduleResult> schedule_result_;
};

static ArrayAttr buildTaskNameArray(MLIRContext *context,
                                    llvm::ArrayRef<TaskflowTaskOp> tasks) {
  llvm::SmallVector<Attribute> task_names;
  for (TaskflowTaskOp task : tasks) {
    task_names.push_back(StringAttr::get(context, task.getTaskName()));
  }
  return ArrayAttr::get(context, task_names);
}

static void emitPipelineIntervalInfo(func::FuncOp func,
                                     const TaskPipelineIntervalResult &result) {
  MLIRContext *context = func.getContext();
  OpBuilder builder(context);

  llvm::SmallVector<NamedAttribute> attrs;
  const int64_t interval = result.pipeline_interval;
  const int64_t max_i32 = std::numeric_limits<int32_t>::max();
  attrs.push_back(builder.getNamedAttr(
      "pipeline_interval", builder.getI32IntegerAttr(static_cast<int32_t>(
                               std::min(interval, max_i32)))));
  if (interval > max_i32) {
    func.emitWarning() << "pipeline interval exceeds i32; "
                          "pipeline_interval_cycles contains the exact value";
    attrs.push_back(builder.getNamedAttr("pipeline_interval_cycles",
                                         builder.getI64IntegerAttr(interval)));
  }

  StringRef bottleneck_task_name = "";
  if (result.bottleneck_task) {
    TaskflowTaskOp bottleneck_task = result.bottleneck_task;
    bottleneck_task_name = bottleneck_task.getTaskName();
  }
  attrs.push_back(builder.getNamedAttr(
      "bottleneck_task", builder.getStringAttr(bottleneck_task_name)));
  attrs.push_back(builder.getNamedAttr(
      "critical_path", buildTaskNameArray(context, result.critical_path)));

  func->setAttr("task_pipeline_interval_info",
                DictionaryAttr::get(context, attrs));
}

struct AnalyzeTaskPipelineIntervalPass
    : public PassWrapper<AnalyzeTaskPipelineIntervalPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(AnalyzeTaskPipelineIntervalPass)

  StringRef getArgument() const override {
    return "analyze-task-pipeline-interval";
  }

  StringRef getDescription() const override {
    return "Analyzes the pipeline interval for scheduled Taskflow tasks";
  }

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    TaskScheduleResultBuilder builder(func);
    FailureOr<llvm::SmallVector<TaskScheduleResult>> schedule_result =
        builder.build();
    if (failed(schedule_result)) {
      signalPassFailure();
      return;
    }

    TaskPipelineIntervalResult result =
        TaskPipelineIntervalAnalyzer(*schedule_result).analyze();
    emitPipelineIntervalInfo(func, result);
  }
};

} // namespace

namespace mlir {
namespace amoeba {
namespace neura {

std::unique_ptr<Pass> createAnalyzeTaskPipelineIntervalPass() {
  return std::make_unique<AnalyzeTaskPipelineIntervalPass>();
}

} // namespace neura
} // namespace amoeba
} // namespace mlir

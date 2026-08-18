# RUN: %python %s %t.linalg.mlir

# RUN: mlir-amoeba-opt %t.linalg.mlir \
# RUN:   --linalg-to-affine-conversion \
# RUN:   -o %t.affine.mlir
# RUN: FileCheck --input-file=%t.affine.mlir %s --check-prefix=AFFINE

# RUN: mlir-amoeba-opt %t.affine.mlir \
# RUN:   --convert-affine-to-taskflow \
# RUN:   -o %t.taskflow.mlir
# RUN: FileCheck --input-file=%t.taskflow.mlir %s --check-prefix=TASKFLOW

# RUN: mlir-amoeba-opt %t.taskflow.mlir \
# RUN:   --construct-hyperblock-from-task \
# RUN:   --cse \
# RUN:   --classify-task-and-counter \
# RUN:   --convert-taskflow-to-neura \
# RUN:   -o %t.neura.mlir
# RUN: FileCheck --input-file=%t.neura.mlir %s --check-prefix=NEURA


# AFFINE:          affine.for %arg1 = 0 to 1 {
# AFFINE:            affine.for %arg2 = 0 to 16 {
# AFFINE:              affine.for %arg3 = 0 to %5 {
# AFFINE:                affine.for %arg4 = 0 to 1 {
# AFFINE-NEXT:             affine.for %arg5 = 0 to 3 {
# AFFINE-NEXT:               %8 = affine.load %arg0[%arg1, %arg4, %arg3 + %arg5] : memref<1x1x?xf32>
# AFFINE-NEXT:               %9 = affine.load %0[%arg2, %arg4, %arg5] : memref<16x1x3xf32>
# AFFINE-NEXT:               %10 = affine.load %alloc[%arg1, %arg2, %arg3] : memref<1x16x?xf32>
# AFFINE-NEXT:               %11 = arith.mulf %8, %9 : f32
# AFFINE-NEXT:               %12 = arith.addf %10, %11 : f32
# AFFINE-NEXT:               affine.store %12, %alloc[%arg1, %arg2, %arg3] : memref<1x16x?xf32>
# AFFINE-NEXT:             }
# AFFINE-NEXT:           }
# AFFINE-NEXT:         }
# AFFINE-NEXT:       }
# AFFINE-NEXT:     }

# TASKFLOW:          %done_writes_0 = taskflow.task @Task_1 will_reads(%arg0, %0, %done_writes : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>) will_writes(%done_writes : memref<1x16x?xf32>) value_inputs(%5 : index) [original_read_memrefs(%arg0, %0, %alloc : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>), original_write_memrefs(%alloc : memref<1x16x?xf32>)] : (memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>, memref<1x16x?xf32>, index) -> (memref<1x16x?xf32>) {
# TASKFLOW-NEXT:     ^bb0(%arg1: memref<1x1x?xf32>, %arg2: memref<16x1x3xf32>, %arg3: memref<1x16x?xf32>, %arg4: memref<1x16x?xf32>, %arg5: index):
# TASKFLOW-NEXT:       affine.for %arg6 = 0 to 1 {
# TASKFLOW-NEXT:         affine.for %arg7 = 0 to 16 {
# TASKFLOW-NEXT:           affine.for %arg8 = 0 to %arg5 {
# TASKFLOW-NEXT:             affine.for %arg9 = 0 to 1 {
# TASKFLOW-NEXT:               affine.for %arg10 = 0 to 3 {
# TASKFLOW-NEXT:                 %8 = affine.load %arg1[%arg6, %arg9, %arg8 + %arg10] : memref<1x1x?xf32>
# TASKFLOW-NEXT:                 %9 = affine.load %arg2[%arg7, %arg9, %arg10] : memref<16x1x3xf32>
# TASKFLOW-NEXT:                 %10 = affine.load %arg4[%arg6, %arg7, %arg8] : memref<1x16x?xf32>
# TASKFLOW-NEXT:                 %11 = arith.mulf %8, %9 : f32
# TASKFLOW-NEXT:                 %12 = arith.addf %10, %11 : f32
# TASKFLOW-NEXT:                 affine.store %12, %arg4[%arg6, %arg7, %arg8] : memref<1x16x?xf32>
# TASKFLOW-NEXT:               }
# TASKFLOW-NEXT:             }
# TASKFLOW-NEXT:           }
# TASKFLOW-NEXT:         }
# TASKFLOW-NEXT:       }
# TASKFLOW-NEXT:       taskflow.yield done_writes(%arg4 : memref<1x16x?xf32>)
# TASKFLOW-NEXT:     }

# NEURA:      %done_writes_0 = taskflow.task @Task_1 will_reads(%arg0, %0, %done_writes : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>) will_writes(%done_writes : memref<1x16x?xf32>) value_inputs(%5 : index) [original_read_memrefs(%arg0, %0, %alloc : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>), original_write_memrefs(%alloc : memref<1x16x?xf32>)] {dlp_replicable = true, runtime_managable = true} : (memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>, memref<1x16x?xf32>, index) -> (memref<1x16x?xf32>) {
# NEURA-NEXT:     ^bb0(%arg1: memref<1x1x?xf32>, %arg2: memref<16x1x3xf32>, %arg3: memref<1x16x?xf32>, %arg4: memref<1x16x?xf32>, %arg5: index):
# NEURA-NEXT:       %c3 = arith.constant 3 : index
# NEURA-NEXT:       %c16 = arith.constant 16 : index
# NEURA-NEXT:       %c0 = arith.constant 0 : index
# NEURA-NEXT:       %c1 = arith.constant 1 : index
# NEURA-NEXT:       %8 = taskflow.counter from %c0 to %c1 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
# NEURA-NEXT:       %9 = taskflow.counter parent(%8 : index) from %c0 to %c16 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
# NEURA-NEXT:       %10 = taskflow.counter parent(%9 : index) from %c0 to %arg5 step %c1 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "relay", counter_id = 2 : i32} : index
# NEURA-NEXT:       %11 = taskflow.counter parent(%10 : index) from %c0 to %c1 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 3 : i32} : index
# NEURA-NEXT:       %12 = taskflow.counter parent(%11 : index) from %c0 to %c3 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 4 : i32} : index
# NEURA-NEXT:       neura.kernel inputs(%arg1, %arg2, %arg4, %arg5 : memref<1x1x?xf32>, memref<16x1x3xf32>, memref<1x16x?xf32>, index) {
# NEURA-NEXT:       ^bb0(%arg6: memref<1x1x?xf32>, %arg7: memref<16x1x3xf32>, %arg8: memref<1x16x?xf32>, %arg9: index):
# NEURA-NEXT:         %c3_15 = arith.constant 3 : index
# NEURA-NEXT:         %c16_16 = arith.constant 16 : index
# NEURA-NEXT:         %c0_17 = arith.constant 0 : index
# NEURA-NEXT:         %c1_18 = arith.constant 1 : index
# NEURA-NEXT:         %13 = neura.counter from %c0_17 : index to %c1_18 : index step %c1_18 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
# NEURA-NEXT:         %14 = neura.counter from %c0_17 : index to %c16_16 : index step %c1_18 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
# NEURA-NEXT:         %15 = neura.counter from %c0_17 : index to %arg9 : index step %c1_18 : index attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "relay", counter_id = 2 : i32} -> index
# NEURA-NEXT:         %16 = neura.counter from %c0_17 : index to %c1_18 : index step %c1_18 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 3 : i32} -> index
# NEURA-NEXT:         %17 = neura.counter from %c0_17 : index to %c3_15 : index step %c1_18 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 4 : i32} -> index
# NEURA-NEXT:         %18 = arith.addi %15, %17 : index
# NEURA-NEXT:         %19 = memref.load %arg6[%13, %16, %18] : memref<1x1x?xf32>
# NEURA-NEXT:         %20 = memref.load %arg7[%14, %16, %17] : memref<16x1x3xf32>
# NEURA-NEXT:         %21 = memref.load %arg8[%13, %14, %15] : memref<1x16x?xf32>
# NEURA-NEXT:         %22 = arith.mulf %19, %20 : f32
# NEURA-NEXT:         %23 = arith.addf %21, %22 : f32
# NEURA-NEXT:         memref.store %23, %arg8[%13, %14, %15] : memref<1x16x?xf32>
# NEURA-NEXT:         neura.yield
# NEURA-NEXT:       }
# NEURA-NEXT:       taskflow.yield done_writes(%arg4 : memref<1x16x?xf32>)
# NEURA-NEXT:     }

"""
Conv1D + Pooling Pipeline — Multi-Task Pipeline with Dynamic time_len

Real-world application:
    - Audio classification (speech command recognition, environmental sound)
    - ECG / biomedical signal analysis (heartbeat anomaly detection)
    - Time-series forecasting (sensor data, financial data)
    Audio clips have variable length: a 1-second clip at 16kHz has 16000
    samples; a 10-second clip has 160000.  ECG segments vary by patient and
    recording duration.  The model is compiled once; runtime adapts to the
    actual signal length.

Multi-task pipeline:
    Task 0-1:  C1 = conv1d(X, F1)     [T, Cin] × [K, Cin, Ch] → [T-K+1, Ch]
    Task 2:    A1 = relu(C1)          [T', Ch]
    Task 3-4:  C2 = conv1d(A1, F2)    [T', Ch] × [K, Ch, Cout] → [T'', Cout]
    Task 5:    A2 = relu(C2)          [T'', Cout]
    Task 6:    P = global_avg_pool(A2) [T'', Cout] → [Cout]
    Task 7-8:  Y = P @ Wfc            [Cout] × [Cout, Ncls] → [Ncls]
    → 9 tasks total, with dynamic time dimension T

    Note: After each conv, the time dimension shrinks by (kernel_size - 1).
    All time-dependent loops have symbol-dynamic bounds derived from T.

Dynamic dimension:
    T (time_len): symbol-dynamic — varies per input signal.
"""

import torch
import torch.nn as nn


class Conv1DPipeline(nn.Module):
    """Conv1D → ReLU → Conv1D → ReLU → GlobalAvgPool → Linear."""

    def __init__(self, in_channels=1, hidden_channels=16,
                 out_channels=32, num_classes=10, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels,
                               kernel_size, bias=False)
        self.conv2 = nn.Conv1d(hidden_channels, out_channels,
                               kernel_size, bias=False)
        self.fc = nn.Linear(out_channels, num_classes, bias=False)

    def forward(self, x):
        # x: [1, in_channels, time_len]  (batch=1)
        h = torch.relu(self.conv1(x))       # [1, Ch, T-K+1]
        h = torch.relu(self.conv2(h))       # [1, Cout, T-2K+2]
        h = torch.mean(h, dim=2)            # [1, Cout]  global avg pool
        return self.fc(h)                    # [1, Ncls]


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(
    out_file,
    time_len=128,
    in_channels=1,
    hidden_channels=16,
    out_channels=32,
    num_classes=10,
    kernel_size=3,
):
    if _try_torch_mlir(out_file, time_len, in_channels, hidden_channels,
                        out_channels, num_classes, kernel_size):
        return
    print("Fail to generate MLIR via torch-mlir.")


def _try_torch_mlir(out_file, T, Cin, Ch, Cout, Ncls, K):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}).")
        return False

    model = Conv1DPipeline(Cin, Ch, Cout, Ncls, K).eval()
    x = torch.randn(1, Cin, T)

    try:
        kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
        from torch.export import Dim
        t_dim = Dim("time", min=2 * K, max=200000)
        kwargs["dynamic_shapes"] = {"x": {2: t_dim}}
        mlir_module = export_and_import(model, x, **kwargs)
        with open(out_file, "w") as f:
            f.write(str(mlir_module))
        print(f"Generated {out_file}  [dynamic shapes via torch-mlir]")
        return True
    except Exception as e:
        print(f"Fail to generate MLIR via torch-mlir ({e})")
    return False


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "conv1d_pipeline_linalg.mlir"
    generate_mlir(out_file=out)

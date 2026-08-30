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

# AFFINE:          %dim_2 = memref.dim %arg1, %c0 : memref<?x?xf32>
# AFFINE-NEXT:     %dim_3 = memref.dim %arg1, %c1 : memref<?x?xf32>
# AFFINE-NEXT:     affine.for %arg2 = 0 to %dim_2 {
# AFFINE-NEXT:       affine.for %arg3 = 0 to 8 {
# AFFINE-NEXT:         affine.for %arg4 = 0 to %dim_3 {
# AFFINE-NEXT:           %4 = affine.load %arg1[%arg2, %arg4] : memref<?x?xf32>
# AFFINE-NEXT:           %5 = affine.load %arg0[%arg4, %arg3] : memref<?x8xf32>
# AFFINE-NEXT:           %6 = affine.load %alloc[%arg2, %arg3] : memref<?x8xf32>
# AFFINE-NEXT:           %7 = arith.mulf %4, %5 : f32
# AFFINE-NEXT:           %8 = arith.addf %6, %7 : f32
# AFFINE-NEXT:           affine.store %8, %alloc[%arg2, %arg3] : memref<?x8xf32>
# AFFINE-NEXT:         }
# AFFINE-NEXT:       }
# AFFINE-NEXT:     }

# TASKFLOW:          %dim_2 = memref.dim %arg1, %c0 : memref<?x?xf32>
# TASKFLOW-NEXT:     %dim_3 = memref.dim %arg1, %c1 : memref<?x?xf32>
# TASKFLOW-NEXT:     %done_writes_4 = taskflow.task @Task_1 will_reads(%arg1, %arg0, %done_writes : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>) will_writes(%done_writes : memref<?x8xf32>) value_inputs(%dim_2, %dim_3 : index, index) [original_read_memrefs(%arg1, %arg0, %alloc : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>), original_write_memrefs(%alloc : memref<?x8xf32>)] : (memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, index, index) -> (memref<?x8xf32>) {
# TASKFLOW-NEXT:     ^bb0(%arg2: memref<?x?xf32>, %arg3: memref<?x8xf32>, %arg4: memref<?x8xf32>, %arg5: memref<?x8xf32>, %arg6: index, %arg7: index):
# TASKFLOW-NEXT:       affine.for %arg8 = 0 to %arg6 {
# TASKFLOW-NEXT:         affine.for %arg9 = 0 to 8 {
# TASKFLOW-NEXT:           affine.for %arg10 = 0 to %arg7 {
# TASKFLOW-NEXT:             %4 = affine.load %arg2[%arg8, %arg10] : memref<?x?xf32>
# TASKFLOW-NEXT:             %5 = affine.load %arg3[%arg10, %arg9] : memref<?x8xf32>
# TASKFLOW-NEXT:             %6 = affine.load %arg5[%arg8, %arg9] : memref<?x8xf32>
# TASKFLOW-NEXT:             %7 = arith.mulf %4, %5 : f32
# TASKFLOW-NEXT:             %8 = arith.addf %6, %7 : f32
# TASKFLOW-NEXT:             affine.store %8, %arg5[%arg8, %arg9] : memref<?x8xf32>
# TASKFLOW-NEXT:           }
# TASKFLOW-NEXT:         }
# TASKFLOW-NEXT:       }
# TASKFLOW-NEXT:       taskflow.yield done_writes(%arg5 : memref<?x8xf32>)
# TASKFLOW-NEXT:     }

# NEURA:          %done_writes_2 = taskflow.task @Task_1 will_reads(%arg1, %arg0, %done_writes : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>) will_writes(%done_writes : memref<?x8xf32>) value_inputs(%dim, %dim_0 : index, index) [original_read_memrefs(%arg1, %arg0, %alloc : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>), original_write_memrefs(%alloc : memref<?x8xf32>)] {dlp_replicable = true, runtime_managable = true} : (memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>, memref<?x8xf32>, index, index) -> (memref<?x8xf32>) {
# NEURA-NEXT:     ^bb0(%arg2: memref<?x?xf32>, %arg3: memref<?x8xf32>, %arg4: memref<?x8xf32>, %arg5: memref<?x8xf32>, %arg6: index, %arg7: index):
# NEURA-NEXT:       %c8 = arith.constant 8 : index
# NEURA-NEXT:       %c0_17 = arith.constant 0 : index
# NEURA-NEXT:       %c1_18 = arith.constant 1 : index
# NEURA-NEXT:       %4 = taskflow.counter from %c0_17 to %arg6 step %c1_18 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
# NEURA-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0_17 to %c8 step %c1_18 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
# NEURA-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0_17 to %arg7 step %c1_18 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
# NEURA-NEXT:       neura.kernel inputs(%arg2, %arg3, %arg5, %arg6, %arg7 : memref<?x?xf32>, memref<?x8xf32>, memref<?x8xf32>, index, index) {
# NEURA-NEXT:       ^bb0(%arg8: memref<?x?xf32>, %arg9: memref<?x8xf32>, %arg10: memref<?x8xf32>, %arg11: index, %arg12: index):
# NEURA-NEXT:         %c8_19 = arith.constant 8 : index
# NEURA-NEXT:         %c0_20 = arith.constant 0 : index
# NEURA-NEXT:         %c1_21 = arith.constant 1 : index
# NEURA-NEXT:         %7 = neura.counter from %c0_20 : index to %arg11 : index step %c1_21 : index attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
# NEURA-NEXT:         %8 = neura.counter from %c0_20 : index to %c8_19 : index step %c1_21 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
# NEURA-NEXT:         %9 = neura.counter from %c0_20 : index to %arg12 : index step %c1_21 : index attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
# NEURA-NEXT:         %10 = memref.load %arg8[%7, %9] : memref<?x?xf32>
# NEURA-NEXT:         %11 = memref.load %arg9[%9, %8] : memref<?x8xf32>
# NEURA-NEXT:         %12 = memref.load %arg10[%7, %8] : memref<?x8xf32>
# NEURA-NEXT:         %13 = arith.mulf %10, %11 : f32
# NEURA-NEXT:         %14 = arith.addf %12, %13 : f32
# NEURA-NEXT:         memref.store %14, %arg10[%7, %8] : memref<?x8xf32>
# NEURA-NEXT:         neura.yield
# NEURA-NEXT:       }
# NEURA-NEXT:       taskflow.yield done_writes(%arg5 : memref<?x8xf32>)
# NEURA-NEXT:     }

"""
Dense 2-Layer GCN — Multi-Task Pipeline with Dynamic N (number of nodes)

Real-world application:
    Graph Neural Networks with dense adjacency are used in:
    - Drug discovery: predicting molecular properties (each molecule is a graph
      with 20-1000 atoms as nodes)
    - Social network analysis: node classification on community subgraphs
    - Protein structure prediction: residue interaction graphs
    The number of nodes N varies per graph instance. A small molecule has ~20
    atoms; a protein has ~1000 residues.  The compiler cannot know N at compile
    time, but N is constant for a given input graph.

Multi-task pipeline (each becomes a separate taskflow.task):
    Task 0-1:  H0 = A @ X          [N, N] × [N, Fin]  → [N, Fin]   aggregate 1
    Task 2-3:  H1 = H0 @ W1        [N, Fin] × [Fin, Fh] → [N, Fh]  combine
    Task 4:    H2 = relu(H1)       [N, Fh] → [N, Fh]               activation
    Task 5-6:  H3 = A @ H2         [N, N] × [N, Fh]  → [N, Fh]     aggregate 2
    Task 7:    out = mean(H3, dim=0) [N, Fh] → [Fh]                 global pool
    → 8 tasks total, all with dynamic N bound

Dynamic dimension:
    N (number of nodes): symbol-dynamic — varies per graph in an inference batch.
"""

import torch
import torch.nn as nn


class DenseTwoLayerGCN(nn.Module):
    """2-layer GCN with dense adjacency: agg → combine → relu → agg → pool."""

    def __init__(self, in_features=8, hidden_features=16):
        super().__init__()
        self.combine = nn.Linear(in_features, hidden_features, bias=False)

    def forward(self, x, adj):
        # x: [N, Fin],  adj: [N, N]  (normalized adjacency)
        h = torch.matmul(adj, x)          # aggregate 1: [N, Fin]
        h = self.combine(h)               # combine:     [N, Fh]
        h = torch.relu(h)                 # activation
        h = torch.matmul(adj, h)          # aggregate 2: [N, Fh]
        out = torch.mean(h, dim=0)        # global pool: [Fh]
        return out


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(
    out_file,
    num_nodes=32,
    in_features=8,
    hidden_features=16,
):
    if _try_torch_mlir(out_file, num_nodes, in_features, hidden_features):
        return
    print("Fail to generate MLIR via torch-mlir.")


def _try_torch_mlir(out_file, N, Fin, Fh):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}).")
        return False

    model = DenseTwoLayerGCN(in_features=Fin, hidden_features=Fh).eval()
    x = torch.randn(N, Fin)
    adj = torch.eye(N) / N

    try:
        kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
        from torch.export import Dim
        n_dim = Dim("n", min=1, max=4096)
        kwargs["dynamic_shapes"] = {
            "x": {0: n_dim},
            "adj": {0: n_dim, 1: n_dim},
        }
        mlir_module = export_and_import(model, x, adj, **kwargs)
        with open(out_file, "w") as f:
            f.write(str(mlir_module))
        print(f"Generated {out_file}  [dynamic shapes via torch-mlir]")
        return True
    except Exception as e:
        print(f"Fail to generate MLIR via torch-mlir ({e})")
    return False

if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "gcn_dynamic_linalg.mlir"
    generate_mlir(out_file=out)

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


# AFFINE:          %dim_4 = memref.dim %arg0, %c0 : memref<?x64xf32>
# AFFINE-NEXT:     affine.for %arg2 = 0 to %dim_4 {
# AFFINE-NEXT:       affine.for %arg3 = 0 to 64 {
# AFFINE-NEXT:         affine.for %arg4 = 0 to 64 {
# AFFINE-NEXT:           %4 = affine.load %arg0[%arg2, %arg4] : memref<?x64xf32>
# AFFINE-NEXT:           %5 = affine.load %alloc[%arg4, %arg3] : memref<64x64xf32>
# AFFINE-NEXT:           %6 = affine.load %alloc_3[%arg2, %arg3] : memref<?x64xf32>
# AFFINE-NEXT:           %7 = arith.mulf %4, %5 : f32
# AFFINE-NEXT:           %8 = arith.addf %6, %7 : f32
# AFFINE-NEXT:           affine.store %8, %alloc_3[%arg2, %arg3] : memref<?x64xf32>
# AFFINE-NEXT:         }
# AFFINE-NEXT:       }
# AFFINE-NEXT:     }

# TASKFLOW:          %done_reads_7, %done_writes_8 = taskflow.task @Task_3 will_reads(%arg0, %done_writes, %done_writes_5 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>) will_writes(%done_writes_5 : memref<?x64xf32>) value_inputs(%dim_6 : index) [original_read_memrefs(%arg0, %alloc, %alloc_4 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>), original_write_memrefs(%alloc_4 : memref<?x64xf32>)] : (memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>, memref<?x64xf32>, index) -> (memref<64x64xf32>, memref<?x64xf32>) {
# TASKFLOW-NEXT:     ^bb0(%arg2: memref<?x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<?x64xf32>, %arg5: memref<?x64xf32>, %arg6: index):
# TASKFLOW-NEXT:       affine.for %arg7 = 0 to %arg6 {
# TASKFLOW-NEXT:         affine.for %arg8 = 0 to 64 {
# TASKFLOW-NEXT:           affine.for %arg9 = 0 to 64 {
# TASKFLOW-NEXT:             %4 = affine.load %arg2[%arg7, %arg9] : memref<?x64xf32>
# TASKFLOW-NEXT:             %5 = affine.load %arg3[%arg9, %arg8] : memref<64x64xf32>
# TASKFLOW-NEXT:             %6 = affine.load %arg5[%arg7, %arg8] : memref<?x64xf32>
# TASKFLOW-NEXT:             %7 = arith.mulf %4, %5 : f32
# TASKFLOW-NEXT:             %8 = arith.addf %6, %7 : f32
# TASKFLOW-NEXT:             affine.store %8, %arg5[%arg7, %arg8] : memref<?x64xf32>
# TASKFLOW-NEXT:           }
# TASKFLOW-NEXT:         }
# TASKFLOW-NEXT:       }
# TASKFLOW-NEXT:       taskflow.yield done_reads(%arg3 : memref<64x64xf32>) done_writes(%arg5 : memref<?x64xf32>)
# TASKFLOW-NEXT:     }

# NEURA:          %done_reads_6, %done_writes_7 = taskflow.task @Task_3 will_reads(%arg0, %done_writes, %done_writes_5 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>) will_writes(%done_writes_5 : memref<?x64xf32>) value_inputs(%dim : index) [original_read_memrefs(%arg0, %alloc, %alloc_4 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>), original_write_memrefs(%alloc_4 : memref<?x64xf32>)] {dlp_replicable = true, runtime_managable = true} : (memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>, memref<?x64xf32>, index) -> (memref<64x64xf32>, memref<?x64xf32>) {
# NEURA-NEXT:     ^bb0(%arg2: memref<?x64xf32>, %arg3: memref<64x64xf32>, %arg4: memref<?x64xf32>, %arg5: memref<?x64xf32>, %arg6: index):
# NEURA-NEXT:       %c64 = arith.constant 64 : index
# NEURA-NEXT:       %c0_45 = arith.constant 0 : index
# NEURA-NEXT:       %c1 = arith.constant 1 : index
# NEURA-NEXT:       %4 = taskflow.counter from %c0_45 to %arg6 step %c1 attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "root", counter_id = 0 : i32} : index
# NEURA-NEXT:       %5 = taskflow.counter parent(%4 : index) from %c0_45 to %c64 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} : index
# NEURA-NEXT:       %6 = taskflow.counter parent(%5 : index) from %c0_45 to %c64 step %c1 attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} : index
# NEURA-NEXT:       neura.kernel inputs(%arg2, %arg3, %arg5, %arg6 : memref<?x64xf32>, memref<64x64xf32>, memref<?x64xf32>, index) {
# NEURA-NEXT:       ^bb0(%arg7: memref<?x64xf32>, %arg8: memref<64x64xf32>, %arg9: memref<?x64xf32>, %arg10: index):
# NEURA-NEXT:         %c64_46 = arith.constant 64 : index
# NEURA-NEXT:         %c0_47 = arith.constant 0 : index
# NEURA-NEXT:         %c1_48 = arith.constant 1 : index
# NEURA-NEXT:         %7 = neura.counter from %c0_47 : index to %arg10 : index step %c1_48 : index attributes {counter_dynamism = "symbol_bound", counter_hierarchy = "root", counter_id = 0 : i32} -> index
# NEURA-NEXT:         %8 = neura.counter from %c0_47 : index to %c64_46 : index step %c1_48 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "relay", counter_id = 1 : i32} -> index
# NEURA-NEXT:         %9 = neura.counter from %c0_47 : index to %c64_46 : index step %c1_48 : index attributes {counter_dynamism = "constant_bound", counter_hierarchy = "leaf", counter_id = 2 : i32} -> index
# NEURA-NEXT:         %10 = memref.load %arg7[%7, %9] : memref<?x64xf32>
# NEURA-NEXT:         %11 = memref.load %arg8[%9, %8] : memref<64x64xf32>
# NEURA-NEXT:         %12 = memref.load %arg9[%7, %8] : memref<?x64xf32>
# NEURA-NEXT:         %13 = arith.mulf %10, %11 : f32
# NEURA-NEXT:         %14 = arith.addf %12, %13 : f32
# NEURA-NEXT:         memref.store %14, %arg9[%7, %8] : memref<?x64xf32>
# NEURA-NEXT:         neura.yield
# NEURA-NEXT:       }
# NEURA-NEXT:       taskflow.yield done_reads(%arg3 : memref<64x64xf32>) done_writes(%arg5 : memref<?x64xf32>)
# NEURA-NEXT:     }

"""
Encoder-Decoder Cross-Attention — Multi-Task Pipeline with Two Dynamic Dims

Real-world application:
    - Machine translation (T5, mBART, NLLB):  source and target sentences
      have *different* and *variable* lengths — "Hello" (5 tokens) translated
      to "Bonjour" (7 tokens).
    - Speech recognition / Whisper:  audio frames (src_len ≈ 1500 for 30 s)
      are attended by generated text tokens (tgt_len varies per utterance).
    - Image captioning (encoder=ViT patches, decoder=caption tokens).

    This benchmark is distinct from self-attention (#1) because Q comes from
    the *target* sequence while K, V come from the *source* sequence.  The
    score matrix is [tgt_len × src_len], mixing two independent dynamic dims.

Multi-task pipeline:
    Task 0-1:   Q  = Tgt @ Wq        [Tt, D] × [D, D]  → [Tt, D]
    Task 2-3:   K  = Src @ Wk        [Ts, D] × [D, D]  → [Ts, D]
    Task 4-5:   V  = Src @ Wv        [Ts, D] × [D, D]  → [Ts, D]
    Task 6-7:   S  = Q @ K^T         [Tt, D] × [D, Ts] → [Tt, Ts]
    Task 8:     A  = softmax(S)      [Tt, Ts]           (omitted in fallback)
    Task 9-10:  Ctx = A @ V          [Tt, Ts] × [Ts, D] → [Tt, D]
    Task 11-12: Out = Ctx @ Wo       [Tt, D] × [D, D]  → [Tt, D]
    → 13 tasks total, src_len and tgt_len are independent dynamic dims.

Dynamic dimensions:
    src_len (Ts): varies per source sentence / audio clip.
    tgt_len (Tt): varies per target sentence / generated output.
    Both are symbol-dynamic — determined once before running the pipeline.
"""

import torch
import torch.nn as nn
import math


class CrossAttention(nn.Module):
    """Encoder-decoder cross-attention: Q from target, K/V from source."""

    def __init__(self, d_model=64):
        super().__init__()
        self.d_model = d_model
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, tgt, src):
        # tgt: [tgt_len, d_model],  src: [src_len, d_model]
        q = self.q_proj(tgt)                                          # [Tt, D]
        k = self.k_proj(src)                                          # [Ts, D]
        v = self.v_proj(src)                                          # [Ts, D]

        scale = 1.0 / math.sqrt(self.d_model)
        scores = torch.matmul(q, k.transpose(0, 1)) * scale          # [Tt, Ts]
        attn = torch.softmax(scores, dim=-1)                          # [Tt, Ts]
        context = torch.matmul(attn, v)                               # [Tt, D]

        return self.out_proj(context)                                 # [Tt, D]


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(out_file,
                  src_len=48, tgt_len=32, d_model=64):
    if _try_torch_mlir(out_file, src_len, tgt_len, d_model):
        return
    print("Fail to generate MLIR via torch-mlir.")


def _try_torch_mlir(out_file, Ts, Tt, D):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}).")
        return False

    model = CrossAttention(D).eval()
    tgt = torch.randn(Tt, D)
    src = torch.randn(Ts, D)

    try:
        kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
        from torch.export import Dim
        ts_dim = Dim("src_len", min=1, max=4096)
        tt_dim = Dim("tgt_len", min=1, max=4096)
        kwargs["dynamic_shapes"] = {
            "tgt": {0: tt_dim},
            "src": {0: ts_dim},
        }
        mlir_module = export_and_import(model, tgt, src, **kwargs)
        with open(out_file, "w") as f:
            f.write(str(mlir_module))
        print(f"Generated {out_file}  [dynamic shapes via torch-mlir]")
        return True
    except Exception as e:
        print(f"Fail to generate MLIR via torch-mlir ({e})")
    return False


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else "cross_attention_linalg.mlir"
    generate_mlir(out_file=out)

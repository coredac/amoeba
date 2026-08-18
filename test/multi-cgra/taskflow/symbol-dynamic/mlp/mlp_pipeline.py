# RUN: %python %s %t.linalg.mlir

# RUN: mlir-amoeba-opt %t.linalg.mlir \
# RUN:   --linalg-to-affine-conversion \
# RUN:   -o %t.affine.mlir
# RUN: FileCheck --input-file=%t.affine.mlir %s --check-prefix=AFFINE

# AFFINE:          %dim_3 = memref.dim %arg0, %c0 : memref<?x64xf32>
# AFFINE-NEXT:     affine.for %arg1 = 0 to %dim_3 {
# AFFINE-NEXT:       affine.for %arg2 = 0 to 128 {
# AFFINE-NEXT:         affine.for %arg3 = 0 to 64 {
# AFFINE-NEXT:           %3 = affine.load %arg0[%arg1, %arg3] : memref<?x64xf32>
# AFFINE-NEXT:           %4 = affine.load %alloc[%arg3, %arg2] : memref<64x128xf32>
# AFFINE-NEXT:           %5 = affine.load %alloc_2[%arg1, %arg2] : memref<?x128xf32>
# AFFINE-NEXT:           %6 = arith.mulf %3, %4 : f32
# AFFINE-NEXT:           %7 = arith.addf %5, %6 : f32
# AFFINE-NEXT:           affine.store %7, %alloc_2[%arg1, %arg2] : memref<?x128xf32>
# AFFINE-NEXT:         }
# AFFINE-NEXT:       }
# AFFINE-NEXT:     }

"""
3-Layer MLP Pipeline — Multi-Task Pipeline with Dynamic batch_size

Real-world application:
    Core compute of recommendation systems (Meta DLRM, Google Wide&Deep),
    tabular data inference, and the dense sub-networks inside larger models.
    In production serving, the batch size varies by traffic: a single user
    click triggers batch=1, while a bulk scoring job uses batch=256.
    The server compiles ONE configuration and the runtime adapts duplication
    based on actual batch size.

Multi-task pipeline:
    Task 0-1:  H1 = X @ W1          [B, D_in] × [D_in, D_h] → [B, D_h]
    Task 2:    A1 = relu(H1)        [B, D_h]
    Task 3-4:  H2 = A1 @ W2         [B, D_h] × [D_h, D_h] → [B, D_h]
    Task 5:    A2 = relu(H2)        [B, D_h]
    Task 6-7:  Y = A2 @ W3          [B, D_h] × [D_h, D_out] → [B, D_out]
    → 8 tasks total, all with dynamic batch dimension B

Dynamic dimension:
    B (batch_size): symbol-dynamic — varies per inference request.
"""

import torch
import torch.nn as nn


class ThreeLayerMLP(nn.Module):
    """3-layer MLP: linear → relu → linear → relu → linear."""

    def __init__(self, d_in=64, d_hidden=128, d_out=32):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=False)
        self.fc2 = nn.Linear(d_hidden, d_hidden, bias=False)
        self.fc3 = nn.Linear(d_hidden, d_out, bias=False)

    def forward(self, x):
        # x: [batch_size, d_in]
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        return self.fc3(h)


# ---------------------------------------------------------------------------
# MLIR generation
# ---------------------------------------------------------------------------

def generate_mlir(
    out_file,
    batch_size=32,
    d_in=64,
    d_hidden=128,
    d_out=32,
):
    if _try_torch_mlir(out_file, batch_size, d_in, d_hidden, d_out):
        return
    print("Fail to generate MLIR via torch-mlir.")


def _try_torch_mlir(out_file, B, d_in, d_hidden, d_out):
    try:
        from torch_mlir.fx import export_and_import
        from torch_mlir.compiler_utils import OutputType
    except ImportError as e:
        print(f"torch-mlir unavailable ({e}).")
        return False

    model = ThreeLayerMLP(d_in, d_hidden, d_out).eval()
    x = torch.randn(B, d_in)

    try:
        kwargs = dict(output_type=OutputType.LINALG_ON_TENSORS, func_name="forward")
        from torch.export import Dim
        batch = Dim("batch", min=1, max=4096)
        kwargs["dynamic_shapes"] = {"x": {0: batch}}
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
    out = sys.argv[1] if len(sys.argv) > 1 else "mlp_linalg.mlir"
    generate_mlir(out_file=out)

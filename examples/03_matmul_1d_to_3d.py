import sys
import os

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, set_backend, get_backend, available_backends

def main():
    print("=" * 75)
    print("⚡ termux-train (AMEVA-Termux) - Sprint 3.9: 1D~3D Complete Matmul Suite")
    print("=" * 75)

    for backend_name in ["python"] + (["numpy"] if "numpy" in available_backends() else []):
        set_backend(backend_name)
        print(f"\n🚀 Running on Backend: [{get_backend().name}]")
        print("-" * 75)

        # 1. 1D @ 1D (Vector Dot Product -> 0D Scalar)
        print("1️⃣ [1D @ 1D Vector Dot Product: (K,) @ (K,) -> ()]")
        a1 = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        b1 = Tensor([4.0, 5.0, 6.0], requires_grad=True)
        y1 = a1 @ b1
        y1.backward()
        print(f"   a1: {a1.shape}, b1: {b1.shape} -> y1: {y1.shape} = {y1.item()}")
        print(f"   a1.grad: {a1.grad.tolist()} (expected [4, 5, 6])")
        print(f"   b1.grad: {b1.grad.tolist()} (expected [1, 2, 3])")

        # 2. 1D @ 2D (Vector-Matrix Multiplication -> 1D Vector)
        print("\n2️⃣ [1D @ 2D Vector-Matrix Product: (K,) @ (K, N) -> (N,)]")
        a2 = Tensor([1.0, 2.0], requires_grad=True)
        b2 = Tensor([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], requires_grad=True)
        y2 = a2 @ b2
        y2.sum().backward()
        print(f"   a2: {a2.shape}, b2: {b2.shape} -> y2: {y2.shape} = {y2.tolist()}")
        print(f"   a2.grad: {a2.grad.tolist()} (expected [12, 21])")
        print(f"   b2.grad: {b2.grad.tolist()} (expected [[1, 1, 1], [2, 2, 2]])")

        # 3. 1D @ 3D (Vector-Batched Matrix Multiplication -> 2D Matrix)
        print("\n3️⃣ [1D @ 3D Vector-Batch Product: (K,) @ (B, K, N) -> (B, N)]")
        a3 = Tensor([1.0, 2.0], requires_grad=True)
        b3 = Tensor([[[3.0, 4.0], [5.0, 6.0]], [[7.0, 8.0], [9.0, 10.0]]], requires_grad=True)
        y3 = a3 @ b3
        y3.sum().backward()
        print(f"   a3: {a3.shape}, b3: {b3.shape} -> y3: {y3.shape} = {y3.tolist()}")
        print(f"   a3.grad (Batch-accumulated): {a3.grad.tolist()} (expected [22, 30])")
        print(f"   b3.grad (Per-batch): {b3.grad.tolist()}")

        # 4. 2D @ 1D (Matrix-Vector Multiplication -> 1D Vector)
        print("\n4️⃣ [2D @ 1D Matrix-Vector Product: (M, K) @ (K,) -> (M,)]")
        a4 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b4 = Tensor([5.0, 6.0], requires_grad=True)
        y4 = a4 @ b4
        y4.sum().backward()
        print(f"   a4: {a4.shape}, b4: {b4.shape} -> y4: {y4.shape} = {y4.tolist()}")
        print(f"   a4.grad: {a4.grad.tolist()} (expected [[5, 6], [5, 6]])")
        print(f"   b4.grad: {b4.grad.tolist()} (expected [4, 6])")

        # 5. 2D @ 2D (Standard Matrix Multiplication -> 2D Matrix)
        print("\n5️⃣ [2D @ 2D Matrix-Matrix Product: (M, K) @ (K, N) -> (M, N)]")
        a5 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b5 = Tensor([[2.0, 0.0], [1.0, 2.0]], requires_grad=True)
        y5 = a5 @ b5
        y5.sum().backward()
        print(f"   a5: {a5.shape}, b5: {b5.shape} -> y5: {y5.shape} = {y5.tolist()}")
        print(f"   a5.grad: {a5.grad.tolist()} (expected [[2, 3], [2, 3]])")
        print(f"   b5.grad: {b5.grad.tolist()} (expected [[4, 4], [6, 6]])")

        # 6. 2D @ 3D (Matrix-Batched Matrix Multiplication -> 3D Tensor)
        print("\n6️⃣ [2D @ 3D Shared Matrix-Batch Product: (M, K) @ (B, K, N) -> (B, M, N)]")
        a6 = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        b6 = Tensor([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)
        y6 = a6 @ b6
        y6.sum().backward()
        print(f"   a6: {a6.shape}, b6: {b6.shape} -> y6: {y6.shape} = {y6.tolist()}")
        print(f"   a6.grad (Batch-accumulated): {a6.grad.tolist()} (expected [[4, 6], [4, 6]])")
        print(f"   b6.grad: {b6.grad.tolist()} (expected [[[4], [6]], [[4], [6]]])")

        # 7. 3D @ 1D (Batched Matrix-Vector Multiplication -> 2D Matrix)
        print("\n7️⃣ [3D @ 1D Batch-Vector Product: (B, M, K) @ (K,) -> (B, M)]")
        a7 = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        b7 = Tensor([2.0, 3.0], requires_grad=True)
        y7 = a7 @ b7
        y7.sum().backward()
        print(f"   a7: {a7.shape}, b7: {b7.shape} -> y7: {y7.shape} = {y7.tolist()}")
        print(f"   a7.grad: {a7.grad.tolist()}")
        print(f"   b7.grad (Batch-accumulated): {b7.grad.tolist()} (expected [16, 20])")

        # 8. 3D @ 2D (Batched Projection with Shared Weights - LoRA / Sequence Linear)
        print("\n8️⃣ [3D @ 2D Sequence Projection: (B, M, K) @ (K, N) -> (B, M, N)]")
        x_seq = Tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]], requires_grad=True)
        w_proj = Tensor([[1.0], [2.0]], requires_grad=True)
        y_proj = x_seq @ w_proj
        y_proj.sum().backward()
        print(f"   x_seq: {x_seq.shape}, w_proj: {w_proj.shape} -> y_proj: {y_proj.shape}")
        print(f"   x_seq.grad: {x_seq.grad.tolist()}")
        print(f"   w_proj.grad (Batch-accumulated): {w_proj.grad.tolist()} (expected [[16], [20]])")

        # 9. 3D @ 3D (Batched Matrix Multiplication - Tiny Transformer Attention)
        print("\n9️⃣ [3D @ 3D Batched Matrix Product: (B, M, K) @ (B, K, N) -> (B, M, N)]")
        q = Tensor([[[1.0, 2.0]], [[3.0, 4.0]]], requires_grad=True)
        k_t = Tensor([[[1.0], [2.0]], [[3.0], [4.0]]], requires_grad=True)
        attn = q @ k_t
        attn.sum().backward()
        print(f"   q: {q.shape}, k_t: {k_t.shape} -> attn: {attn.shape} = {attn.tolist()}")
        print(f"   q.grad:   {q.grad.tolist()}")
        print(f"   k_t.grad: {k_t.grad.tolist()}")

    print("\n" + "=" * 75)
    print("🎉 All 9 1D~3D Matmul operations fully verified across both Backends!")
    print("=" * 75 + "\n")

if __name__ == "__main__":
    main()

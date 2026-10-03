"""
termux_train.vision.trainer
===========================
End-to-End On-Device Vision VLM LoRA Trainer.
Manages dataset iteration, forward multimodal projection, cross-entropy backprop,
and automated export to termux-vision and HuggingFace SafeTensors format.
Open-Source under Apache License 2.0.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List

import termux_train as tt
from termux_train import Tensor, get_backend, set_backend
from termux_train.nn.loss import cross_entropy_loss
from termux_train.optim.adamw import AdamW
from termux_train.data.vision_dataset import VisionQADataset
from termux_train.vision.vlm_lora import VisionLanguageModelLoRA
from termux_train.adapters.hub import export_target_adapter, TargetEcosystem

logger = logging.getLogger(__name__)


def train_vision_vlm_lora(
    data_source: str,
    output_path: str,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 1,
    vocab_size: int = 1000,
    visual_dim: int = 64,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    rank: int = 4,
    alpha: float = 1.0,
    max_seq_len: int = 32,
    backend: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes end-to-end on-device Vision VLM LoRA training loop.

    Returns:
        Summary dict containing training metrics, epoch losses, and output adapter path.
    """
    if backend and backend != "auto":
        set_backend(backend)
    b = get_backend()

    if verbose:
        print("=" * 65)
        print("  👁️ AMEVA On-Device Vision VLM LoRA Training Pipeline")
        print("=" * 65)
        print(f"  • Data Source     : {data_source}")
        print(f"  • Output Adapter  : {output_path}")
        print(f"  • Compute Backend : {b.name.upper()}")
        print(f"  • VLM Dimensions  : VisualDim={visual_dim} -> LLMDim={d_model} (Heads={num_heads})")
        print(f"  • LoRA Config     : Rank={rank}, Alpha={alpha}, LR={lr}")
        print("=" * 65)

    # 1. Load Dataset
    t0_data = time.perf_counter()
    dataset = VisionQADataset(
        data_source=data_source,
        patch_dim=visual_dim,
        num_patches=16,
        max_seq_len=max_seq_len,
        vocab_size=vocab_size,
    )
    if verbose:
        print(f"  📦 Loaded {len(dataset)} VLM QA samples in {(time.perf_counter() - t0_data)*1000:.1f}ms")

    # 2. Initialize VLM with LoRA
    model = VisionLanguageModelLoRA(
        vocab_size=vocab_size,
        visual_dim=visual_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        rank=rank,
        alpha=alpha,
        backend=b,
    )

    # Collect trainable LoRA parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if verbose:
        print(f"  🔧 Trainable LoRA Parameters: {len(trainable_params)} tensors")

    optimizer = AdamW(trainable_params, lr=lr)

    history = []
    t_start_train = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps = 0
        t0_epoch = time.perf_counter()

        for idx in range(0, len(dataset), batch_size):
            # Batch extraction
            batch_vis = []
            batch_in = []
            batch_tgt = []

            for b_i in range(idx, min(idx + batch_size, len(dataset))):
                vis_feat, in_ids, tgt_ids = dataset[b_i]
                batch_vis.append(vis_feat)
                batch_in.append(in_ids)
                batch_tgt.append(tgt_ids)

            # Stack into batch tensors
            # visual_features: (B, N, D)
            B = len(batch_vis)
            N = batch_vis[0].shape[0]
            D = batch_vis[0].shape[1]
            vis_all = []
            for vf in batch_vis:
                vis_all.extend(b.to_flat_list(vf._data))
            vis_tensor = Tensor(b.from_data(b.reshape(vis_all, (B, N, D)), dtype="float32"), dtype="float32", backend=b)

            # in_ids: (B, S), tgt_ids: (B, S)
            S = batch_in[0].shape[0]
            in_all = []
            tgt_all = []
            for inp, tgt in zip(batch_in, batch_tgt):
                in_all.extend(b.to_flat_list(inp._data))
                tgt_all.extend(b.to_flat_list(tgt._data))

            in_tensor = Tensor(b.from_data(b.reshape(in_all, (B, S)), dtype="int64"), dtype="int64", backend=b)
            tgt_tensor = Tensor(b.from_data(b.reshape(tgt_all, (B, S)), dtype="int64"), dtype="int64", backend=b)

            # Forward
            optimizer.zero_grad()
            logits = model(vis_tensor, in_tensor)

            # Cross-Entropy Loss
            loss = cross_entropy_loss(logits, tgt_tensor)
            loss_val = float(loss.item())

            # Autograd Backward
            loss.backward()

            # Optimizer Step
            optimizer.step()

            epoch_loss += loss_val
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        dur = (time.perf_counter() - t0_epoch) * 1000.0
        history.append({"epoch": epoch, "loss": avg_loss, "duration_ms": dur})
        if verbose:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Time: {dur:.1f}ms")

    total_time = time.perf_counter() - t_start_train

    # 3. Export to Target-Aware SafeTensors Adapter Hub
    abs_out = export_target_adapter(
        model_or_adapters=model,
        target=TargetEcosystem.VISION,
        output_path=output_path,
        base_model_name="termux-vlm-base",
        custom_metadata={
            "epochs": str(epochs),
            "final_loss": f"{history[-1]['loss']:.6f}" if history else "0.0",
            "visual_dim": str(visual_dim),
            "llm_dim": str(d_model),
            "rank": str(rank),
        },
    )

    if verbose:
        print("=" * 65)
        print(f"  ✅ VLM LoRA Training Complete: {epochs} epochs in {total_time:.2f}s")
        print(f"  💾 Exported Adapter: {abs_out}")
        print("=" * 65)

    return {
        "status": "success",
        "epochs": epochs,
        "final_loss": history[-1]["loss"] if history else 0.0,
        "adapter_path": abs_out,
        "history": history,
        "total_time_sec": total_time,
    }

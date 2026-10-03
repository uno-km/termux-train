"""
termux_train.audio.trainer
==========================
End-to-End On-Device Audio STT & TTS LoRA Trainers.
Optimizes Whisper acoustic cross-attention and TTS speaker adaptation,
with automated export to termux-stt / termux-tts SafeTensors formats.
Open-Source under Apache License 2.0.
"""

import os
import time
import logging
from typing import Optional, Dict, Any, List

import termux_train as tt
from termux_train import Tensor, get_backend, set_backend
from termux_train.nn.loss import cross_entropy_loss, mse_loss
from termux_train.optim.adamw import AdamW
from termux_train.data.audio_dataset import AudioTranscriptionDataset, AudioVoiceDataset
from termux_train.audio.audio_lora import WhisperSTTLoRAModel, TTSSpeakerAdaptationLoRA
from termux_train.adapters.hub import export_target_adapter, TargetEcosystem

logger = logging.getLogger(__name__)


def train_stt_lora(
    data_source: str,
    output_path: str,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 1,
    vocab_size: int = 512,
    audio_dim: int = 80,
    d_model: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    rank: int = 4,
    alpha: float = 1.0,
    max_text_len: int = 32,
    backend: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes on-device Whisper STT Cross-Attention LoRA acoustic training.
    """
    if backend and backend != "auto":
        set_backend(backend)
    b = get_backend()

    if verbose:
        print("=" * 65)
        print("  🎙️ AMEVA On-Device Audio STT (Whisper) LoRA Training Pipeline")
        print("=" * 65)
        print(f"  • Data Source     : {data_source}")
        print(f"  • Output Adapter  : {output_path}")
        print(f"  • Compute Backend : {b.name.upper()}")
        print(f"  • Acoustic Dims   : AudioDim={audio_dim} -> DecoderDim={d_model}")
        print(f"  • LoRA Config     : Rank={rank}, Alpha={alpha}, LR={lr}")
        print("=" * 65)

    # 1. Dataset
    t0_data = time.perf_counter()
    dataset = AudioTranscriptionDataset(
        data_source=data_source,
        n_mels=audio_dim,
        num_frames=32,
        max_text_len=max_text_len,
        vocab_size=vocab_size,
    )
    if verbose:
        print(f"  📦 Loaded {len(dataset)} Audio STT samples in {(time.perf_counter() - t0_data)*1000:.1f}ms")

    # 2. Model
    model = WhisperSTTLoRAModel(
        vocab_size=vocab_size,
        audio_dim=audio_dim,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        rank=rank,
        alpha=alpha,
        backend=b,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr)

    history = []
    t_start_train = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps = 0
        t0_epoch = time.perf_counter()

        for idx in range(0, len(dataset), batch_size):
            batch_mel = []
            batch_in = []
            batch_tgt = []

            for b_i in range(idx, min(idx + batch_size, len(dataset))):
                mel_feat, in_ids, tgt_ids = dataset[b_i]
                batch_mel.append(mel_feat)
                batch_in.append(in_ids)
                batch_tgt.append(tgt_ids)

            B = len(batch_mel)
            T_audio = batch_mel[0].shape[0]
            D_audio = batch_mel[0].shape[1]
            mel_all = []
            for mf in batch_mel:
                mel_all.extend(b.to_flat_list(mf._data))
            mel_tensor = Tensor(b.from_data(b.reshape(mel_all, (B, T_audio, D_audio)), dtype="float32"), dtype="float32", backend=b)

            S = batch_in[0].shape[0]
            in_all = []
            tgt_all = []
            for inp, tgt in zip(batch_in, batch_tgt):
                in_all.extend(b.to_flat_list(inp._data))
                tgt_all.extend(b.to_flat_list(tgt._data))

            in_tensor = Tensor(b.from_data(b.reshape(in_all, (B, S)), dtype="int64"), dtype="int64", backend=b)
            tgt_tensor = Tensor(b.from_data(b.reshape(tgt_all, (B, S)), dtype="int64"), dtype="int64", backend=b)

            optimizer.zero_grad()
            logits = model(mel_tensor, in_tensor)
            loss = cross_entropy_loss(logits, tgt_tensor)
            loss_val = float(loss.item())

            loss.backward()
            optimizer.step()

            epoch_loss += loss_val
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        dur = (time.perf_counter() - t0_epoch) * 1000.0
        history.append({"epoch": epoch, "loss": avg_loss, "duration_ms": dur})
        if verbose:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Time: {dur:.1f}ms")

    total_time = time.perf_counter() - t_start_train

    # 3. Export to Target-Aware SafeTensors Hub
    abs_out = export_target_adapter(
        model_or_adapters=model,
        target=TargetEcosystem.STT,
        output_path=output_path,
        base_model_name="whisper-tiny-acoustic",
        custom_metadata={
            "epochs": str(epochs),
            "final_loss": f"{history[-1]['loss']:.6f}" if history else "0.0",
            "audio_dim": str(audio_dim),
            "rank": str(rank),
        },
    )

    if verbose:
        print("=" * 65)
        print(f"  ✅ Audio STT LoRA Training Complete in {total_time:.2f}s")
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


def train_tts_lora(
    data_source: str,
    output_path: str,
    epochs: int = 5,
    lr: float = 0.001,
    batch_size: int = 1,
    vocab_size: int = 512,
    n_mels: int = 80,
    d_model: int = 64,
    rank: int = 4,
    alpha: float = 1.0,
    backend: str = "auto",
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Executes on-device TTS Speaker Adaptation LoRA training.
    """
    if backend and backend != "auto":
        set_backend(backend)
    b = get_backend()

    if verbose:
        print("=" * 65)
        print("  🗣️ AMEVA On-Device Audio TTS (Voice Style) LoRA Training Pipeline")
        print("=" * 65)
        print(f"  • Data Source     : {data_source}")
        print(f"  • Output Adapter  : {output_path}")
        print(f"  • Compute Backend : {b.name.upper()}")
        print(f"  • Prosody Target  : MelBins={n_mels}, ModelDim={d_model}")
        print(f"  • LoRA Config     : Rank={rank}, Alpha={alpha}, LR={lr}")
        print("=" * 65)

    # 1. Dataset
    t0_data = time.perf_counter()
    dataset = AudioVoiceDataset(data_source=data_source, n_mels=n_mels, num_frames=32)
    if verbose:
        print(f"  📦 Loaded {len(dataset)} Voice Style samples in {(time.perf_counter() - t0_data)*1000:.1f}ms")

    # 2. Model
    model = TTSSpeakerAdaptationLoRA(
        vocab_size=vocab_size,
        d_model=d_model,
        n_mels=n_mels,
        rank=rank,
        alpha=alpha,
        backend=b,
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=lr)

    history = []
    t_start_train = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps = 0
        t0_epoch = time.perf_counter()

        for idx in range(0, len(dataset), batch_size):
            batch_in = []
            batch_mel = []

            for b_i in range(idx, min(idx + batch_size, len(dataset))):
                in_ids, target_mel = dataset[b_i]
                batch_in.append(in_ids)
                batch_mel.append(target_mel)

            B = len(batch_in)
            S = batch_in[0].shape[0]
            in_all = []
            for inp in batch_in:
                in_all.extend(b.to_flat_list(inp._data))
            in_tensor = Tensor(b.from_data(b.reshape(in_all, (B, S)), dtype="int64"), dtype="int64", backend=b)

            # Match frames: text length S to mel target frames
            optimizer.zero_grad()
            pred_mels = model(in_tensor)  # (B, S, n_mels)

            # Target slice: slice target_mel to S frames
            mel_slices = []
            for tm in batch_mel:
                flat_tm = b.to_flat_list(tm._data)
                # Take first S * n_mels
                sliced = flat_tm[:S * n_mels]
                if len(sliced) < S * n_mels:
                    sliced.extend([0.0] * (S * n_mels - len(sliced)))
                mel_slices.extend(sliced)

            target_tensor = Tensor(b.from_data(b.reshape(mel_slices, (B, S, n_mels)), dtype="float32"), dtype="float32", backend=b)

            loss = mse_loss(pred_mels, target_tensor)
            loss_val = float(loss.item())

            loss.backward()
            optimizer.step()

            epoch_loss += loss_val
            steps += 1

        avg_loss = epoch_loss / max(1, steps)
        dur = (time.perf_counter() - t0_epoch) * 1000.0
        history.append({"epoch": epoch, "loss": avg_loss, "duration_ms": dur})
        if verbose:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.4f} | Time: {dur:.1f}ms")

    total_time = time.perf_counter() - t_start_train

    # 3. Export to Target-Aware SafeTensors Hub
    abs_out = export_target_adapter(
        model_or_adapters=model,
        target=TargetEcosystem.TTS,
        output_path=output_path,
        base_model_name="termux-tts-speaker-base",
        custom_metadata={
            "epochs": str(epochs),
            "final_loss": f"{history[-1]['loss']:.6f}" if history else "0.0",
            "n_mels": str(n_mels),
            "rank": str(rank),
        },
    )

    if verbose:
        print("=" * 65)
        print(f"  ✅ Audio TTS LoRA Training Complete in {total_time:.2f}s")
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

"""
Tests for on-device Audio STT & TTS LoRA Training Pipelines.
Verifies AudioTranscriptionDataset, AudioVoiceDataset, Whisper STT LoRA,
TTS Speaker Adaptation LoRA, Loss convergence, SafeTensors export, and CLI integration.
"""

import os
import json
import tempfile
import pytest

import termux_train as tt
from termux_train import Tensor
from termux_train.data.audio_dataset import AudioTranscriptionDataset, AudioVoiceDataset
from termux_train.audio.audio_lora import WhisperSTTLoRAModel, TTSSpeakerAdaptationLoRA
from termux_train.audio.trainer import train_stt_lora, train_tts_lora
from termux_train.checkpoint.safetensors import load_safetensors


@pytest.fixture
def temp_audio_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(3):
            wav_p = os.path.join(tmpdir, f"audio_{i}.wav")
            with open(wav_p, "wb") as f:
                f.write(f"RIFF_SYNTHETIC_WAV_{i}".encode("utf-8") * 10)
            
            txt_p = os.path.join(tmpdir, f"audio_{i}.txt")
            with open(txt_p, "w", encoding="utf-8") as f:
                f.write(f"synthetic voice transcript number {i}")

        yield tmpdir


def test_audio_datasets_and_caching(temp_audio_dir):
    ds_stt = AudioTranscriptionDataset(
        data_source=temp_audio_dir,
        n_mels=32,
        num_frames=16,
        max_text_len=16,
        vocab_size=100,
    )
    assert len(ds_stt) == 3

    mel, in_ids, tgt_ids = ds_stt[0]
    assert mel.shape == (16, 32)
    assert in_ids.ndim == 1
    assert tgt_ids.ndim == 1
    assert in_ids.shape[0] == tgt_ids.shape[0] == 15

    # Check cache
    cache_files = os.listdir(ds_stt.cache_dir)
    assert len(cache_files) >= 1

    ds_tts = AudioVoiceDataset(
        data_source=temp_audio_dir,
        n_mels=32,
        num_frames=16,
    )
    assert len(ds_tts) == 3
    in_text, target_mel = ds_tts[0]
    assert target_mel.shape == (16, 32)


def test_whisper_stt_lora_forward_backward():
    vocab_size = 100
    audio_dim = 32
    d_model = 16
    num_heads = 2
    rank = 2

    model = WhisperSTTLoRAModel(
        vocab_size=vocab_size,
        audio_dim=audio_dim,
        d_model=d_model,
        num_layers=1,
        num_heads=num_heads,
        rank=rank,
        alpha=1.0,
    )

    # Base frozen, LoRA trainable
    assert model.tok_emb.weight.requires_grad is False
    assert model.head.weight.requires_grad is False
    assert model.cross_attns[0].q_proj.lora_A.requires_grad is True

    b = model.tok_emb.weight.backend
    B, T_audio, S_text = 2, 8, 4
    mel = Tensor(b.from_data(b.reshape([0.2] * (B * T_audio * audio_dim), (B, T_audio, audio_dim)), dtype="float32"), dtype="float32", backend=b)
    txt_ids = Tensor(b.from_data(b.reshape([3] * (B * S_text), (B, S_text)), dtype="int64"), dtype="int64", backend=b)
    tgt_ids = Tensor(b.from_data(b.reshape([4] * (B * S_text), (B, S_text)), dtype="int64"), dtype="int64", backend=b)

    logits = model(mel, txt_ids)
    assert logits.shape == (B, S_text, vocab_size)

    from termux_train.nn.loss import cross_entropy_loss
    loss = cross_entropy_loss(logits, tgt_ids)
    assert float(loss.item()) > 0.0

    loss.backward()
    assert model.cross_attns[0].q_proj.lora_A.grad is not None


def test_tts_speaker_adaptation_lora_forward_backward():
    vocab_size = 100
    d_model = 16
    n_mels = 32
    rank = 2

    model = TTSSpeakerAdaptationLoRA(
        vocab_size=vocab_size,
        d_model=d_model,
        n_mels=n_mels,
        rank=rank,
        alpha=1.0,
    )

    b = model.tok_emb.weight.backend
    B, S = 2, 8
    txt_ids = Tensor(b.from_data(b.reshape([2] * (B * S), (B, S)), dtype="int64"), dtype="int64", backend=b)
    target_mel = Tensor(b.from_data(b.reshape([0.1] * (B * S * n_mels), (B, S, n_mels)), dtype="float32"), dtype="float32", backend=b)

    out_mel = model(txt_ids)
    assert out_mel.shape == (B, S, n_mels)

    from termux_train.nn.loss import mse_loss
    loss = mse_loss(out_mel, target_mel)
    assert float(loss.item()) > 0.0

    loss.backward()
    assert model.speaker_adapter1.lora_A.grad is not None


def test_train_stt_and_tts_lora_end_to_end(temp_audio_dir):
    out_stt = os.path.join(temp_audio_dir, "test_stt_adapter.safetensors")
    res_stt = train_stt_lora(
        data_source=temp_audio_dir,
        output_path=out_stt,
        epochs=2,
        lr=0.01,
        batch_size=1,
        vocab_size=100,
        audio_dim=16,
        d_model=16,
        num_layers=1,
        num_heads=2,
        rank=2,
        max_text_len=8,
        backend="auto",
        verbose=False,
    )
    assert res_stt["status"] == "success"
    assert os.path.isfile(out_stt)
    loaded_stt, _ = load_safetensors(out_stt)
    assert any("stt.acoustic." in k for k in loaded_stt.keys())

    out_tts = os.path.join(temp_audio_dir, "test_tts_adapter.safetensors")
    res_tts = train_tts_lora(
        data_source=temp_audio_dir,
        output_path=out_tts,
        epochs=2,
        lr=0.01,
        batch_size=1,
        vocab_size=100,
        n_mels=16,
        d_model=16,
        rank=2,
        backend="auto",
        verbose=False,
    )
    assert res_tts["status"] == "success"
    assert os.path.isfile(out_tts)
    loaded_tts, _ = load_safetensors(out_tts)
    assert any("tts.speaker." in k for k in loaded_tts.keys())


def test_cli_audio_train_args():
    from termux_train.cli import main
    import sys
    for subcmd in ("stt-train", "tts-train"):
        test_args = ["termux-train", subcmd, "--help"]
        with pytest.raises(SystemExit) as exc_info:
            sys.argv = test_args
            main()
        assert exc_info.value.code == 0

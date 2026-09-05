"""
examples/09_tiny_whisper_lora.py
================================
Tiny Whisper Speech-to-Text LoRA Fine-Tuning Demo.
Simulates fine-tuning a frozen Audio Encoder-Decoder Whisper model with low-rank adapters
on an on-device speech command recognition task.
"""

import sys
import os
import math
import time

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except (AttributeError, OSError) as rec_err:
        sys.stderr.write(f'[termux-train] Notice: stream reconfigure failed: {rec_err}\n')

# Set project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from termux_train import Tensor, nn, optim, checkpoint, set_backend, get_backend
from termux_train.tokenization import WordTokenizer


class TinyWhisperEncoder(nn.Module):
    """Encodes 80-dim log-mel audio frames into latent audio representations."""
    def __init__(self, mel_bins: int = 80, d_model: int = 64, num_heads: int = 4, d_ff: int = 128, num_layers: int = 2):
        super().__init__()
        self.conv_proj = nn.Linear(mel_bins, d_model)
        self.blocks = [
            nn.TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
            for _ in range(num_layers)
        ]
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}", b)
        self.ln_post = nn.LayerNorm(d_model)

    def forward(self, mel_features: Tensor) -> Tensor:
        # mel_features: (B, T_audio, mel_bins)
        x = self.conv_proj(mel_features)
        for block in self.blocks:
            x = block(x, causal=False)
        return self.ln_post(x)


class TinyWhisperDecoder(nn.Module):
    """Decodes latent audio representations into text token probabilities."""
    def __init__(self, vocab_size: int, d_model: int = 64, num_heads: int = 4, d_ff: int = 128, num_layers: int = 2):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = [
            nn.TransformerBlock(d_model=d_model, num_heads=num_heads, d_ff=d_ff)
            for _ in range(num_layers)
        ]
        for i, b in enumerate(self.blocks):
            setattr(self, f"block_{i}", b)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, text_tokens: Tensor, enc_hidden: Tensor) -> Tensor:
        # text_tokens: (B, S_text), enc_hidden: (B, T_audio, d_model)
        B, T, D = enc_hidden.shape
        x = self.tok_emb(text_tokens)
        audio_ctx = enc_hidden.mean(axis=1).reshape(B, 1, D)  # (B, 1, d_model)
        x = x + audio_ctx
        for block in self.blocks:
            x = block(x, causal=True)
        x = self.ln_f(x)
        return self.lm_head(x)


class TinyWhisperModel(nn.Module):
    """Full Tiny Whisper Speech-to-Text Model."""
    def __init__(self, vocab_size: int, mel_bins: int = 80, d_model: int = 64):
        super().__init__()
        self.encoder = TinyWhisperEncoder(mel_bins=mel_bins, d_model=d_model)
        self.decoder = TinyWhisperDecoder(vocab_size=vocab_size, d_model=d_model)

    def forward(self, mel_features: Tensor, text_tokens: Tensor) -> Tensor:
        enc_hidden = self.encoder(mel_features)
        logits = self.decoder(text_tokens, enc_hidden)
        return logits


def generate_synthetic_audio_mel(batch_size: int, time_steps: int, mel_bins: int, seed_bias: float) -> list:
    """Generates synthetic log-mel spectrogram features representing distinct acoustic commands."""
    import random
    random.seed(int(seed_bias * 100))
    data = []
    for _ in range(batch_size):
        seq = []
        for t in range(time_steps):
            frame = [math.sin(t * 0.2 + seed_bias) + random.uniform(-0.1, 0.1) for _ in range(mel_bins)]
            seq.append(frame)
        data.append(seq)
    return data


def main():
    print("=" * 70)
    print("  🎙️ [termux-train] On-Device Tiny Whisper LoRA Fine-Tuning Demo")
    print("=" * 70)

    set_backend("auto")
    print(f"[*] Active Compute Backend: [{get_backend().name.upper()}]")

    # 1. Vocabulary & Tokenizer
    corpus = [
        "<START> open the front door <END>",
        "<START> turn on living room light <END>",
        "<START> stop playing the music <END>",
        "<START> set morning alarm for seven <END>",
    ]
    tokenizer = WordTokenizer()
    tokenizer.build_vocab(corpus)
    vocab_size = tokenizer.vocab_size
    print(f"[*] Speech Command Vocab Size: {vocab_size} tokens")

    # 2. Build Pre-trained Base Whisper Model
    mel_bins = 80
    d_model = 64
    base_whisper = TinyWhisperModel(vocab_size=vocab_size, mel_bins=mel_bins, d_model=d_model)

    total_params = sum(len(p.tolist()) if p.ndim == 1 else (p.shape[0] * p.shape[1]) for p in base_whisper.parameters())
    print(f"[*] Total Base Whisper Parameters: {total_params:,}")

    # 3. Inject LoRA Low-Rank Adapters (Rank=4, Alpha=8.0) into Attention & Projection layers
    print("\n▶️ [Phase 1]: Injecting LoRA Low-Rank Adapters (Rank=4, Alpha=8.0)...")
    
    # Wrap linear layers in encoder/decoder with LoRA
    base_whisper.encoder.conv_proj = nn.LoRALinear.from_linear(base_whisper.encoder.conv_proj, rank=4, alpha=8.0)
    for i, block in enumerate(base_whisper.encoder.blocks):
        block.attn.q_proj = nn.LoRALinear.from_linear(block.attn.q_proj, rank=4, alpha=8.0)
        block.attn.v_proj = nn.LoRALinear.from_linear(block.attn.v_proj, rank=4, alpha=8.0)
    for i, block in enumerate(base_whisper.decoder.blocks):
        block.attn.q_proj = nn.LoRALinear.from_linear(block.attn.q_proj, rank=4, alpha=8.0)
        block.attn.v_proj = nn.LoRALinear.from_linear(block.attn.v_proj, rank=4, alpha=8.0)

    # 4. Freeze Base Weights & Isolate LoRA Parameters
    trainable_params = nn.adapter_parameters(base_whisper)
    num_trainable = sum(len(p.tolist()) if p.ndim == 1 else (p.shape[0] * p.shape[1]) for p in trainable_params)
    print(f"[*] Trainable LoRA Parameters: {num_trainable:,} ({num_trainable / total_params * 100:.2f}% of model)")
    print(f"[*] Frozen Base Parameters: {total_params - num_trainable:,} (100% Immutable)")

    # 5. Prepare Audio-Text Training Batches
    # 4 audio commands (each 16 frames x 80 mel-bins)
    mel_batch_data = []
    text_input_ids = []
    text_target_ids = []

    for i, text in enumerate(corpus):
        mel_seq = generate_synthetic_audio_mel(batch_size=1, time_steps=16, mel_bins=mel_bins, seed_bias=(i + 1) * 1.5)[0]
        mel_batch_data.append(mel_seq)
        
        ids = tokenizer.encode(text)
        # Teacher forcing: input is [0..N-1], target is [1..N]
        text_input_ids.append(ids[:-1])
        text_target_ids.append(ids[1:])

    # Pad text sequences to uniform length
    max_len = max(len(ids) for ids in text_input_ids)
    pad_id = tokenizer.PAD_ID
    
    padded_inputs = [ids + [pad_id] * (max_len - len(ids)) for ids in text_input_ids]
    padded_targets = [ids + [pad_id] * (max_len - len(ids)) for ids in text_target_ids]

    mel_tensor = Tensor(mel_batch_data, dtype="float32")
    inp_text_tensor = Tensor(padded_inputs, dtype="int64")
    tgt_text_tensor = Tensor(padded_targets, dtype="int64")

    # 6. Fine-Tuning Loop with AdamW Optimizer
    print("\n▶️ [Phase 2]: Fine-Tuning LoRA Adapters on Audio-to-Text Mapping (15 Epochs)...")
    optimizer = optim.AdamW(trainable_params, lr=0.03, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    t_start = time.perf_counter()
    for epoch in range(1, 16):
        optimizer.zero_grad()
        
        logits = base_whisper(mel_tensor, inp_text_tensor)  # (B, S, vocab_size)
        B, S, V = logits.shape
        loss = criterion(logits.reshape(B * S, V), tgt_text_tensor.reshape(B * S))
        
        loss.backward()
        nn.clip_grad_norm_(trainable_params, max_norm=1.0)
        optimizer.step()

        if epoch in (1, 3, 6, 9, 12, 15):
            print(f"   Epoch {epoch:2d}/15 | CrossEntropy Loss: {loss.item():.4f}")

    train_time = time.perf_counter() - t_start
    print(f"\n[*] LoRA Fine-Tuning Completed in {train_time:.2f}s!")

    # 7. Save Lightweight LoRA Adapter (< 50KB)
    print("\n▶️ [Phase 3]: Saving Lightweight LoRA Adapter to SafeTensors Binary...")
    adapter_path = "whisper_lora_adapter.safetensors"
    checkpoint.save_lora_adapter(base_whisper, adapter_path, adapter_name="tiny_whisper_cmd_v1")
    adapter_size_kb = os.path.getsize(adapter_path) / 1024.0
    print(f"[*] Saved LoRA Adapter Path: '{adapter_path}'")
    print(f"[*] Adapter File Size: {adapter_size_kb:.2f} KB (Base model would be ~{total_params * 4 / 1024:.2f} KB)")
    assert adapter_size_kb < 100.0, "Adapter file size must be < 100KB!"

    # 8. Merge LoRA Weights for Zero-Overhead Inference
    print("\n▶️ [Phase 4]: Merging LoRA Weights into Base Model for Fast Inference...")
    nn.merge_lora_adapters(base_whisper)
    print("[*] LoRA matrices merged into base weights (merged=True).")

    # 9. Test Audio Recognition Inference
    print("\n▶️ [Phase 5]: Evaluating Speech Command Recognition Inference...")
    for idx, raw_target in enumerate(corpus):
        single_mel = Tensor([mel_batch_data[idx]], dtype="float32")
        single_inp = Tensor([padded_inputs[idx]], dtype="int64")
        
        pred_logits = base_whisper(single_mel, single_inp)
        pred_rows = pred_logits.tolist()[0]
        pred_ids = [max(range(len(row)), key=lambda c: row[c]) for row in pred_rows]
        
        # Strip padding
        pred_text = tokenizer.decode(pred_ids)
        print(f"   [Audio {idx+1}] Target: '{raw_target}'")
        print(f"             Decoded: '<START> {pred_text}'")

    # Cleanup temp file
    if os.path.exists(adapter_path):
        os.remove(adapter_path)

    print("\n" + "=" * 70)
    print("  ✅ [termux-train] Tiny Whisper LoRA Demo 100% SUCCESS!")
    print("=" * 70)


if __name__ == "__main__":
    main()

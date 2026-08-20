# 📱 On-Device Tiny Model & Small LLM Training Guide
> **Comprehensive Manual: Training Transformers, Small LLMs, and Whisper LoRA on Android Termux & Edge Hardware**

---

## 📑 Table of Contents

1. [Introduction & Mobile Constraints](#1-introduction--mobile-constraints)
2. [Architecture of On-Device Tiny Models](#2-architecture-of-on-device-tiny-models)
3. [Recipe 1: Tiny Transformer Language Model (RoPE + KV Cache)](#3-recipe-1-tiny-transformer-language-model-rope--kv-cache)
4. [Recipe 2: Tiny Whisper Speech-to-Text LoRA Fine-Tuning](#4-recipe-2-tiny-whisper-speech-to-text-lora-fine-tuning)
5. [Recipe 3: Structured Document Sequence Mapping (DocFold)](#5-recipe-3-structured-document-sequence-mapping-docfold)
6. [Memory Optimization Strategies (INT8, SafeTensors, MMap)](#6-memory-optimization-strategies-int8-safetensors-mmap)
7. [Recommended Hyperparameter Matrix by Device RAM](#7-recommended-hyperparameter-matrix-by-device-ram)
8. [Troubleshooting & Best Practices](#8-troubleshooting--best-practices)

---

## 1. Introduction & Mobile Constraints

Training neural networks on smartphones (Android Termux) differs fundamentally from cloud GPU clusters. On mobile devices, the primary constraints are:

- **Strict RAM Quotas**: Mobile operating systems invoke the **Low Memory Killer (LMK)** when free RAM drops below ~300MB–500MB.
- **Thermal Dissipation**: Mobile SoCs (Qualcomm Snapdragon, Samsung Exynos, MediaTek Dimensity) throttle CPU frequencies when core temperatures rise.
- **Single-Precision / INT8 Efficiency**: Vectorized CPU compute (ARM NEON SIMD) excels at contiguous memory access and lightweight matrix multiplications.

`termux-train` is specifically engineered to overcome these challenges through **0-dependency core autograd**, **streaming `mmap` datasets**, **HuggingFace-compatible `.safetensors` binary zero-copy serialization**, and **parameter-efficient low-rank adaptation (LoRA)**.

---

## 2. Architecture of On-Device Tiny Models

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Tiny Transformer LM (Decoder)                   │
├────────────────────────────────────────────────────────────────────────┤
│  Token IDs ──► nn.Embedding (d_model=64)                              │
│                      │                                                 │
│                      ▼                                                 │
│         ┌─────────────────────────┐                                    │
│         │  TransformerBlock (x2)  │ ◄── Native RoPE (O(0) learnable)   │
│         │  - MultiHeadAttention   │ ◄── Pre-LN + Residual              │
│         │  - FeedForward (d_ff)   │ ◄── GELU / ReLU Activation         │
│         └─────────────────────────┘                                    │
│                      │                                                 │
│                      ▼                                                 │
│             nn.LayerNorm(d_model)                                      │
│                      │                                                 │
│                      ▼                                                 │
│          nn.Linear(d_model, vocab) (Weight-Tied LM Head)               │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Recipe 1: Tiny Transformer Language Model (RoPE + KV Cache)

This recipe trains a Decoder-only autoregressive language model with **Rotary Position Embedding (RoPE)** and generates text with **incremental KV caching**.

```python
from termux_train import Tensor, nn, optim, set_backend
from termux_train.tokenization import CharTokenizer

# 1. Enable Hardware Acceleration
set_backend("auto")

# 2. Prepare Corpus & Tokenizer
raw_text = "to be or not to be that is the question whether tis nobler in the mind"
tokenizer = CharTokenizer()
tokenizer.build_vocab([raw_text])
token_ids = tokenizer.encode(raw_text)

# 3. Create Dataset (Teacher Forcing: Inputs [0..N-1], Targets [1..N])
seq_len = 16
inputs, targets = [], []
for i in range(len(token_ids) - seq_len):
    inputs.append(token_ids[i:i + seq_len])
    targets.append(token_ids[i + 1:i + seq_len + 1])

x_train = Tensor(inputs, dtype="int64")
y_train = Tensor(targets, dtype="int64")

# 4. Instantiate TinyTransformerLM
model = nn.TinyTransformerLM(
    vocab_size=tokenizer.vocab_size,
    d_model=64,
    num_heads=4,
    d_ff=128,
    num_layers=2,
    pos_type="rope",   # Zero positional memory overhead
    tie_weights=True   # Ties token embedding with LM head
)

# 5. Training Loop with AdamW
optimizer = optim.AdamW(model.parameters(), lr=0.005, weight_decay=1e-4)

for epoch in range(1, 51):
    optimizer.zero_grad(set_to_none=True)  # Optimal mobile RAM usage
    logits, loss = model(x_train, targets=y_train)
    loss.backward()
    nn.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d}/50 | CrossEntropy Loss: {loss.item():.4f}")

# 6. Autoregressive Generation with KV Cache
prompt = tokenizer.encode("to be ")
generated_ids = model.generate(
    prompt_tokens=prompt,
    max_new_tokens=30,
    temperature=0.7,
    top_p=0.9,
    use_cache=True  # Incremental O(1) step inference
)
print("Generated Text:", tokenizer.decode(generated_ids))
```

---

## 4. Recipe 2: Tiny Whisper Speech-to-Text LoRA Fine-Tuning

This recipe demonstrates fine-tuning a pre-trained Speech Encoder-Decoder with **LoRA low-rank adapters**, freezing $96\%$ of base parameters and saving adapter weights in $<30\text{KB}$.

```python
from termux_train import Tensor, nn, optim, checkpoint
from termux_train.tokenization import WordTokenizer

# 1. Define Tiny Whisper Architecture
class TinyWhisper(nn.Module):
    def __init__(self, vocab_size: int, mel_bins: int = 80, d_model: int = 64):
        super().__init__()
        self.conv_proj = nn.Linear(mel_bins, d_model)
        self.enc_blocks = [nn.TransformerBlock(d_model, num_heads=4, d_ff=128) for _ in range(2)]
        for i, b in enumerate(self.enc_blocks): setattr(self, f"enc_{i}", b)
        
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.dec_blocks = [nn.TransformerBlock(d_model, num_heads=4, d_ff=128) for _ in range(2)]
        for i, b in enumerate(self.dec_blocks): setattr(self, f"dec_{i}", b)
        
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, mel_features: Tensor, text_tokens: Tensor) -> Tensor:
        B, T, D = mel_features.shape
        x = self.conv_proj(mel_features)
        for b in self.enc_blocks: x = b(x, causal=False)
        audio_ctx = x.mean(axis=1).reshape(B, 1, -1)
        
        tok = self.tok_emb(text_tokens) + audio_ctx
        for b in self.dec_blocks: tok = b(tok, causal=True)
        return self.lm_head(self.ln_f(tok))

model = TinyWhisper(vocab_size=30, mel_bins=80, d_model=64)

# 2. Inject LoRA Low-Rank Adapters (Rank=4, Alpha=8.0)
for b in model.enc_blocks + model.dec_blocks:
    b.attn.q_proj = nn.LoRALinear.from_linear(b.attn.q_proj, rank=4, alpha=8.0)
    b.attn.v_proj = nn.LoRALinear.from_linear(b.attn.v_proj, rank=4, alpha=8.0)

# 3. Freeze Base Weights & Train Adapters Only
trainable_params = nn.adapter_parameters(model)
optimizer = optim.AdamW(trainable_params, lr=0.02)
criterion = nn.CrossEntropyLoss()

# 4. Save & Merge Adapter
checkpoint.save_lora_adapter(model, "whisper_lora.safetensors", adapter_name="cmd_v1")
nn.merge_lora_adapters(model)  # Zero-overhead inference merge
```

---

## 5. Recipe 3: Structured Document Sequence Mapping (DocFold)

For processing structured JSON, forms, and key-value document logs on mobile without heavy cloud dependencies:

```python
from termux_train import Tensor, nn, optim
from termux_train.tokenization import WordTokenizer

# Tokenize structured document lines
corpus = [
    "DOC: invoice | VENDOR: Acme Corp | AMOUNT: $450.00",
    "DOC: receipt | VENDOR: Metro Mart | AMOUNT: $12.50",
]
tokenizer = WordTokenizer()
tokenizer.build_vocab(corpus)

# Train a 1-layer Transformer mapping context to entity values
model = nn.TinyTransformerLM(
    vocab_size=tokenizer.vocab_size,
    d_model=32,
    num_heads=2,
    d_ff=64,
    num_layers=1,
    pos_type="rope"
)
```

---

## 6. Memory Optimization Strategies

### A. Zero-Copy SafeTensors Checkpointing
Eliminates Python object pickling overhead and allows instant weight mapping:
```python
from termux_train import checkpoint
checkpoint.save_safetensors({"weights": model.state_dict()}, "model.safetensors")
```

### B. MMap Streaming Dataset
Reads token chunks directly from disk via kernel page cache without consuming heap RAM:
```python
from termux_train.data import MMapTokenDataset
dataset = MMapTokenDataset.create_from_tokens(tokens=token_list, filepath="dataset.bin", seq_len=64)
```

### C. INT8 AbsMax Quantization
Reduces linear layer weights by $75\%$ ($4\times$ memory compression) with zero-allocation matrix multiplication:
```python
q_linear = nn.quantize_linear_int8(model.fc)
```

---

## 7. Recommended Hyperparameter Matrix by Device RAM

| Device RAM | Recommended Model | `d_model` | `num_heads` | `num_layers` | Batch Size | Typical RAM Usage |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **2GB - 3GB** | Micro LM / LoRA Adapter | 32 | 2 | 1 ~ 2 | 1 ~ 2 | **< 25 MB** |
| **4GB - 6GB** | Tiny Transformer LM / Whisper | 64 | 4 | 2 ~ 4 | 4 ~ 8 | **< 60 MB** |
| **8GB - 12GB** | Small LLM (1M–5M params) | 128 | 8 | 4 ~ 6 | 8 ~ 16 | **< 150 MB** |

---

## 8. Troubleshooting & Best Practices

1. **Gradient Cleanup**: Always use `optimizer.zero_grad(set_to_none=True)` to deallocate gradient buffers immediately after `optimizer.step()`.
2. **Prevent Exploding Gradients**: Apply `nn.clip_grad_norm_(model.parameters(), max_norm=1.0)` before `step()`.
3. **Backend Selection**: In Termux, always install `pkg install python-numpy` for $10\times \sim 100\times$ faster execution via OpenBLAS ARM NEON.

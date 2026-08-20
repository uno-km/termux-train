"""
examples/05_transformer_lm.py
==============================
Character-Level Autoregressive Language Model Demo for Mobile/Termux.

Demonstrates:
  1. Lightweight CharTokenizer fitting on small text corpus
  2. Pure Python / NumPy TinyTransformerLM training with AdamW and CrossEntropyLoss
  3. Real-time autoregressive text continuation generation
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, nn, optim, set_backend, available_backends
from termux_train.tokenization import CharTokenizer


def main():
    print("=" * 70)
    print("  🚀 [termux-train] Character-Level Transformer LM Demo (Mobile-First)")
    print("=" * 70)

    # 1. Select fast backend if available
    backends = available_backends()
    chosen_backend = "numpy" if "numpy" in backends else "python"
    set_backend(chosen_backend)
    print(f"[*] Active Backend: {chosen_backend.upper()}")

    # 2. Prepare Sample Corpus
    corpus = (
        "To be, or not to be, that is the question: "
        "Whether 'tis nobler in the mind to suffer "
        "The slings and arrows of outrageous fortune, "
        "Or to take arms against a sea of troubles "
        "And by opposing end them. To die—to sleep, "
        "No more; and by a sleep to say we end "
        "The heart-ache and the thousand natural shocks "
        "That flesh is heir to: 'tis a consummation "
        "Devoutly to be wish'd."
    )
    print(f"[*] Raw Corpus Length: {len(corpus)} characters")

    # 3. Fit Tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab([corpus])
    vocab_size = tokenizer.vocab_size
    print(f"[*] Vocab Size: {vocab_size} unique characters")

    token_ids = tokenizer.encode(corpus)
    print(f"[*] Encoded Tokens Count: {len(token_ids)}")

    # 4. Model Hyperparameters (Optimized for Mobile RAM & Instant Execution)
    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2
    seq_len = 16
    batch_size = 4
    learning_rate = 0.01
    epochs = 15

    model = nn.TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=seq_len + 1
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"[*] Model Architecture: {model}")
    total_params = sum(len(p.data) if hasattr(p.data, "__len__") else 1 for p in model.parameters())
    print(f"[*] Model Parameters Registered: {len(list(model.parameters()))}")

    # 5. Build Training Batches (Sliding Windows)
    inputs_list = []
    targets_list = []
    for i in range(0, len(token_ids) - seq_len - 1, seq_len // 2):
        inp = token_ids[i:i + seq_len]
        tgt = token_ids[i + 1:i + seq_len + 1]
        inputs_list.append(inp)
        targets_list.append(tgt)

    print(f"[*] Prepared {len(inputs_list)} Sequence Windows (seq_len={seq_len})")

    # 6. Training Loop
    print("\n--- Training Progress ---")
    start_time = time.time()
    initial_loss = 0.0
    final_loss = 0.0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        batches_run = 0

        for b in range(0, len(inputs_list), batch_size):
            b_inps = inputs_list[b:b + batch_size]
            b_tgts = targets_list[b:b + batch_size]
            if len(b_inps) == 0:
                continue

            x = Tensor(b_inps, dtype="int64")
            y = Tensor(b_tgts, dtype="int64")

            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()

            loss_val = float(loss.item())
            epoch_loss += loss_val
            batches_run += 1

        avg_loss = epoch_loss / max(1, batches_run)
        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

        if epoch % 3 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\n[*] Training Complete in {elapsed:.2f}s! Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")

    # 7. Autoregressive Text Generation Demo
    prompt_text = "To be"
    prompt_tokens = tokenizer.encode(prompt_text)
    print(f"\n--- Autoregressive Generation Demo ---")
    print(f"[*] Prompt: '{prompt_text}'")

    generated_ids = model.generate(prompt_tokens, max_new_tokens=30, temperature=0.8)
    generated_text = tokenizer.decode(generated_ids)
    print(f"[*] Generated: '{generated_text}'\n")

    print("=" * 70)
    print("  ✅ [termux-train] Demo 05 Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

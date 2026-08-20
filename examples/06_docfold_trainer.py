"""
examples/06_docfold_trainer.py
==============================
DocFold Sequence Mapping Toy Trainer for Termux Mobile Devices.

Demonstrates:
  1. Loading unstructured document texts from DocFoldDataset
  2. Training TinyTransformerLM to map input text into structured symbols (<DOC>, <HEADER>, <VALUE>, <END>)
  3. Overfitting / convergence on mobile-friendly toy records
  4. Real-time inference and structured symbol generation
"""

import os
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from termux_train import Tensor, nn, optim, set_backend, available_backends
from termux_train.data import DocFoldDataset, DocFoldRecord


def main():
    print("=" * 70)
    print("  📑 [termux-train] DocFold Sequence Mapping Toy Trainer")
    print("=" * 70)

    # 1. Select Backend
    backends = available_backends()
    chosen_backend = "numpy" if "numpy" in backends else "python"
    set_backend(chosen_backend)
    print(f"[*] Active Backend: {chosen_backend.upper()}")

    # 2. Prepare Dataset
    dataset = DocFoldDataset.create_toy_dataset()
    tokenizer = dataset.tokenizer
    vocab_size = tokenizer.vocab_size
    print(f"[*] Dataset Records: {len(dataset)}")
    print(f"[*] WordTokenizer Vocab Size: {vocab_size}")

    for idx, r in enumerate(dataset):
        print(f"    [{idx + 1}] Raw: '{r.raw_text}' -> Target: '{r.to_symbolic_sequence()}'")

    # 3. Model & Optimizer Configuration
    d_model = 32
    num_heads = 4
    d_ff = 64
    num_layers = 2
    max_seq_len = 32
    epochs = 20
    learning_rate = 0.015

    model = nn.TinyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        max_seq_len=max_seq_len
    )
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    print(f"\n[*] Model: {model}")

    # 4. Generate Batches
    batches = dataset.create_batches(batch_size=2, max_seq_len=max_seq_len, ignore_index=-100)
    print(f"[*] Generated {len(batches)} Training Batches (batch_size=2, max_seq_len={max_seq_len})")

    # 5. Training Loop
    print("\n--- Training Progress ---")
    start_time = time.time()
    initial_loss = 0.0
    final_loss = 0.0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        for x, y in batches:
            optimizer.zero_grad()
            logits, loss = model(x, targets=y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())

        avg_loss = epoch_loss / len(batches)
        if epoch == 1:
            initial_loss = avg_loss
        final_loss = avg_loss

        if epoch % 4 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | Loss: {avg_loss:.4f}")

    elapsed = time.time() - start_time
    print(f"\n[*] Training Complete in {elapsed:.2f}s! Initial Loss: {initial_loss:.4f} -> Final Loss: {final_loss:.4f}")

    # 6. Structured Mapping Inference Demo
    test_raw = "Invoice #1042 Total: $450 Vendor: Acme Corp"
    prompt_str = f"{test_raw} -> <DOC>"
    prompt_tokens = tokenizer.encode(prompt_str, add_bos=True, add_eos=False)

    print("\n--- Structured Inference Demo ---")
    print(f"[*] Input Raw Text: '{test_raw}'")
    print(f"[*] Prompt Given: '{prompt_str}'")

    generated_ids = model.generate(prompt_tokens, max_new_tokens=15, temperature=0.7)
    decoded_output = tokenizer.decode(generated_ids)
    print(f"[*] Generated Structured Output:\n    '{decoded_output}'\n")

    print("=" * 70)
    print("  ✅ [termux-train] Demo 06 Completed Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

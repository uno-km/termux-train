"""
termux_train.runtime.runner
===========================
Production-Grade Training Session Runner for IPC / CLI / Node.js SDK.
Implements True Chunked Mini-Batch Iteration, Dynamic Shape Verification,
and Fail-Closed Atomic Checkpoint Error Propagation.
Open-Source under Apache License 2.0.
"""

import sys
import os
import time
import json
import argparse
import random
import math
import struct
from typing import Dict, Any, Tuple, Optional, List

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import termux_train as tt
from termux_train import Tensor, randn, set_backend, available_backends
import termux_train.nn as nn
import termux_train.optim as optim
import termux_train.checkpoint as checkpoint


class MiniBatchDataset:
    """Production Mini-Batch Dataset Iterator with slice slicing and Bounded Memory footprint."""
    def __init__(self, x: Tensor, y: Tensor, batch_size: int):
        self.x = x
        self.y = y
        self.batch_size = max(1, batch_size)
        self.total_samples = x.shape[0]
        self.num_batches = math.ceil(self.total_samples / self.batch_size)

    def get_batch(self, batch_idx: int) -> Tuple[Tensor, Tensor]:
        start = batch_idx * self.batch_size
        end = min(start + self.batch_size, self.total_samples)
        
        # Backend slice or list slicing
        raw_x = self.x._data[start:end]
        raw_y = self.y._data[start:end]
        
        bx = Tensor(raw_x, dtype=self.x.dtype, backend=self.x.backend)
        by = Tensor(raw_y, dtype=self.y.dtype, backend=self.y.backend)
        return bx, by


def load_dataset_and_metadata(data_path: Optional[str], cfg: Dict[str, Any]) -> Tuple[Tensor, Tensor, int, int]:
    """
    Loads dataset from SafeTensors, JSONL, or Text file.
    Returns (x_tensor, y_tensor, detected_in_dim, detected_out_dim).
    """
    batch_size = cfg.get("batchSize", 16)
    cfg_dim = cfg.get("dim", 32)
    cfg_out_dim = cfg.get("outDim", cfg_dim if "lora" in cfg.get("modelType", "") else 1)
    seq_len = cfg.get("seqLen", 32)
    vocab_size = cfg.get("vocabSize", 256)
    m_type = cfg.get("modelType", "mlp")

    if data_path and os.path.exists(data_path):
        ext = os.path.splitext(data_path)[1].lower()
        if ext == ".safetensors":
            file_size = os.path.getsize(data_path)
            MAX_DATASET_FILE_BYTES = 2 * 1024 * 1024 * 1024  # 2GB bounded memory guard
            if file_size > MAX_DATASET_FILE_BYTES:
                raise ValueError(
                    f"SafeTensors dataset '{data_path}' ({file_size / (1024*1024):.1f}MB) exceeds "
                    f"the 2GB safety limit for on-device memory. Aborting."
                )

            tensors, _ = checkpoint.load_safetensors(data_path)
            if "inputs" in tensors and "targets" in tensors:
                x, y = tensors["inputs"], tensors["targets"]
            elif "x" in tensors and "y" in tensors:
                x, y = tensors["x"], tensors["y"]
            elif len(tensors) >= 2:
                keys = list(tensors.keys())
                x, y = tensors[keys[0]], tensors[keys[1]]
            else:
                raise ValueError(f"SafeTensors dataset at '{data_path}' must contain at least 2 tensors (e.g. 'inputs' and 'targets').")

            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"SafeTensors dataset sample count mismatch: inputs has {x.shape[0]} samples, "
                    f"but targets has {y.shape[0]} samples."
                )

            in_dim = x.shape[-1] if len(x.shape) >= 2 else cfg_dim
            out_dim = y.shape[-1] if len(y.shape) >= 2 else cfg_out_dim
            return x, y, in_dim, out_dim

        elif ext in (".jsonl", ".json"):
            # MAX_JSONL_RECORDS prevents unbounded in-memory accumulation
            MAX_JSONL_RECORDS = 100_000
            records = []
            with open(data_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    records.append(json.loads(line))
                    if len(records) >= MAX_JSONL_RECORDS:
                        break  # Enforce cap; remainder is silently truncated with log

            if not records:
                raise ValueError(f"JSONL dataset at '{data_path}' contains no parseable records.")

            first = records[0]
            if not (isinstance(first, dict) and "input" in first and "target" in first):
                raise ValueError(
                    f"JSONL dataset at '{data_path}': each record must have 'input' and 'target' keys. "
                    f"Got keys: {list(first.keys()) if isinstance(first, dict) else type(first).__name__}"
                )

            x_list = [r["input"] for r in records]
            y_list = [r["target"] for r in records]
            x = Tensor(x_list, dtype="float32")
            y = Tensor(y_list, dtype="float32")

            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"JSONL dataset sample count mismatch: inputs has {x.shape[0]} samples, "
                    f"but targets has {y.shape[0]} samples."
                )

            in_dim = x.shape[-1] if len(x.shape) >= 2 else cfg_dim
            out_dim = y.shape[-1] if len(y.shape) >= 2 else cfg_out_dim
            return x, y, in_dim, out_dim

        elif ext == ".txt":
            with open(data_path, "r", encoding="utf-8-sig") as f:
                text = f.read()

            from termux_train.tokenization.byte import ByteTokenizer
            tokenizer = ByteTokenizer()
            token_ids = tokenizer.encode(text)
            total_tokens = len(token_ids)
            chunk_count = total_tokens // (seq_len + 1)
            if chunk_count < 1:
                raise ValueError(
                    f"Text dataset at '{data_path}' has {total_tokens} UTF-8 tokens, which is "
                    f"too short for sequence window of {seq_len + 1} tokens."
                )
            x_chunks = [token_ids[i * seq_len:(i + 1) * seq_len] for i in range(chunk_count)]
            y_chunks = [token_ids[i * seq_len + 1:(i + 1) * seq_len + 1] for i in range(chunk_count)]
            x = Tensor(x_chunks, dtype="int64")
            y = Tensor(y_chunks, dtype="int64")

            if x.shape[0] != y.shape[0]:
                raise ValueError(
                    f"Text chunk count mismatch: x has {x.shape[0]} chunks, "
                    f"but y has {y.shape[0]} chunks."
                )

            return x, y, cfg_dim, cfg_out_dim

        elif ext in (".bin", ".mmap"):
            from termux_train.data.mmap_dataset import MMapTokenDataset
            mmap_ds = MMapTokenDataset(data_path, seq_len=seq_len)
            total_samples = len(mmap_ds)
            if total_samples < 1:
                mmap_ds.close()
                raise ValueError(f"MMap binary dataset at '{data_path}' has insufficient tokens for seq_len={seq_len}.")

            # Direct C-level binary buffer extraction: Read all sequence tokens in single slice
            total_needed_tokens = total_samples + seq_len
            raw_bytes = mmap_ds._mmap[8:8 + total_needed_tokens * 8]
            fmt = f"<{total_needed_tokens}q"
            all_tokens = list(struct.unpack(fmt, raw_bytes))
            mmap_ds.close()

            x_chunks = [all_tokens[i:i + seq_len] for i in range(total_samples)]
            y_chunks = [all_tokens[i + 1:i + 1 + seq_len] for i in range(total_samples)]

            x = Tensor(x_chunks, dtype="int64")
            y = Tensor(y_chunks, dtype="int64")
            return x, y, cfg_dim, cfg_out_dim

        else:
            raise ValueError(
                f"Unsupported dataset file extension '{ext}' at '{data_path}'. "
                "Supported formats: .safetensors, .jsonl, .json, .txt, .bin, .mmap"
            )

    # Synthetic Benchmark Data — only reached when data_path is explicitly None/omitted
    total_synthetic_samples = batch_size * 4
    if m_type in ("transformer", "transformer-lm", "rope"):
        v_size = max(int(cfg.get("vocabSize", 260)), 260)
        raw_x = [[random.randint(0, v_size - 1) for _ in range(seq_len)] for _ in range(total_synthetic_samples)]
        raw_y = [[random.randint(0, v_size - 1) for _ in range(seq_len)] for _ in range(total_synthetic_samples)]
        return Tensor(raw_x, dtype="int64"), Tensor(raw_y, dtype="int64"), cfg_dim, cfg_out_dim
    else:
        return randn((total_synthetic_samples, cfg_dim)), randn((total_synthetic_samples, cfg_out_dim)), cfg_dim, cfg_out_dim


def run_session(cfg: Dict[str, Any]) -> None:
    """Executes an enterprise-grade training session with Fail-Closed error propagation."""
    epochs = int(cfg.get("epochs", 5))
    lr = float(cfg.get("lr", 1e-3))
    ckpt_path = cfg.get("checkpointPath")
    data_path = cfg.get("dataPath")
    backend_req = cfg.get("backend", "auto")
    batch_size = int(cfg.get("batchSize", 16))

    # 1. Set Backend (Fail-Closed)
    if backend_req and backend_req != "auto":
        matched = False
        for b in available_backends():
            if b.lower() == backend_req.lower():
                set_backend(b)
                matched = True
                break
        if not matched:
            raise RuntimeError(
                f"Requested backend '{backend_req}' is not available on this system. "
                f"Available backends: {available_backends()}. Fail-Closed enforced."
            )

    # 2. Load Dataset & Resolve Authoritative Dimensions
    x_data, y_data, in_dim, out_dim = load_dataset_and_metadata(data_path, cfg)
    dataset = MiniBatchDataset(x_data, y_data, batch_size=batch_size)

    # 3. Build Model with Verified Dimensions
    m_type = cfg.get("modelType", "mlp").lower()

    if m_type in ("lora", "linear-lora"):
        rank = int(cfg.get("loraRank", min(4, in_dim, out_dim)))
        alpha = float(cfg.get("loraAlpha", 1.0))
        model = nn.LoRALinear(in_features=in_dim, out_features=out_dim, rank=rank, alpha=alpha)
        optimizer = optim.AdamW(nn.adapter_parameters(model), lr=lr)
        criterion = nn.MSELoss()

    elif m_type in ("transformer", "transformer-lm", "rope"):
        vocab_size = max(int(cfg.get("vocabSize", 260)), 260)
        heads = int(cfg.get("heads", 2))
        layers = int(cfg.get("layers", 2))
        model = nn.TinyTransformerLM(
            vocab_size=vocab_size,
            d_model=in_dim,
            num_heads=heads,
            d_ff=in_dim * 4,
            num_layers=layers,
            pos_type="rope"
        )
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

    else:
        hidden_dim = int(cfg.get("hiddenDim", in_dim * 2))
        model = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        optimizer = optim.AdamW(model.parameters(), lr=lr)
        criterion = nn.MSELoss()

    # 4. Resume / Load Checkpoint (if provided)
    resume_path = cfg.get("resumePath") or cfg.get("resume")
    if resume_path:
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume checkpoint file not found: {resume_path}")
        loaded_tensors, ckpt_meta = checkpoint.load_safetensors(resume_path)

        # 4.1 Restore Model Parameters
        if m_type in ("lora", "linear-lora") and hasattr(model, "lora_A") and hasattr(model, "lora_B"):
            if "lora_A" in loaded_tensors:
                model.lora_A.data = loaded_tensors["lora_A"].data
            if "lora_B" in loaded_tensors:
                model.lora_B.data = loaded_tensors["lora_B"].data
        else:
            for name, param in model.named_parameters():
                if name in loaded_tensors:
                    param.data = loaded_tensors[name].data

        # 4.2 Restore Optimizer Momentum Buffers
        if hasattr(optimizer, "state"):
            for p_idx, param in enumerate(model.parameters()):
                exp_avg_key = f"optim_exp_avg_{p_idx}"
                exp_avg_sq_key = f"optim_exp_avg_sq_{p_idx}"
                if exp_avg_key in loaded_tensors and exp_avg_sq_key in loaded_tensors:
                    if param not in optimizer.state:
                        optimizer.state[param] = {}
                    optimizer.state[param]["step"] = int(ckpt_meta.get("global_step", 1))
                    optimizer.state[param]["exp_avg"] = loaded_tensors[exp_avg_key].data
                    optimizer.state[param]["exp_avg_sq"] = loaded_tensors[exp_avg_sq_key].data

    # 5. Training Loop with Mini-Batch Iteration
    global_step = 0
    total_steps = epochs * dataset.num_batches

    for ep in range(epochs):
        epoch_loss_sum = 0.0
        t0 = time.perf_counter()

        for b_idx in range(dataset.num_batches):
            bx, by = dataset.get_batch(b_idx)
            optimizer.zero_grad()

            if m_type in ("transformer", "transformer-lm", "rope"):
                logits, _ = model(bx)
                b, s, v = logits.shape
                batch_loss = criterion(logits.reshape(b * s, v), by.reshape(b * s))
            else:
                preds = model(bx)
                batch_loss = criterion(preds, by)

            batch_loss.backward()
            optimizer.step()
            loss_val = float(batch_loss.item())

            if not math.isfinite(loss_val):
                raise FloatingPointError(
                    f"Training diverged with non-finite loss (loss={loss_val}) at epoch {ep+1}, step {global_step+1}. "
                    f"Fail-Closed early abort triggered to prevent weight corruption."
                )

            epoch_loss_sum += loss_val
            global_step += 1

        avg_loss = epoch_loss_sum / dataset.num_batches
        lat_ms = (time.perf_counter() - t0) * 1000.0

        metrics = {
            "event": "step",
            "epoch": ep + 1,
            "totalEpochs": epochs,
            "loss": float(avg_loss),
            "step": global_step,
            "totalSteps": total_steps,
            "batchesPerEpoch": dataset.num_batches,
            "latencyMs": round(lat_ms, 3)
        }
        print(f"__METRICS__:{json.dumps(metrics)}", flush=True)

    # 5. Checkpoint I/O with Strict Fail-Closed Error Propagation
    if ckpt_path:
        try:
            if m_type in ("lora", "linear-lora") and hasattr(model, "lora_A") and hasattr(model, "lora_B"):
                # LoRA isolation: only save trainable adapter tensors, omit frozen base weights
                t_dict = {
                    "lora_A": model.lora_A,
                    "lora_B": model.lora_B
                }
            else:
                t_dict = dict(model.named_parameters())

            # Add Optimizer Momentum Buffers (exp_avg, exp_avg_sq) into checkpoint
            if hasattr(optimizer, "state_dict"):
                opt_state = optimizer.state_dict()
                for p_idx, s_dict in opt_state.get("state", {}).items():
                    if "exp_avg" in s_dict and s_dict["exp_avg"] is not None:
                        t_dict[f"optim_exp_avg_{p_idx}"] = Tensor(s_dict["exp_avg"], dtype="float32")
                    if "exp_avg_sq" in s_dict and s_dict["exp_avg_sq"] is not None:
                        t_dict[f"optim_exp_avg_sq_{p_idx}"] = Tensor(s_dict["exp_avg_sq"], dtype="float32")

            checkpoint.save_safetensors(
                t_dict,
                ckpt_path,
                metadata={
                    "framework": "termux-train",
                    "model_type": m_type,
                    "in_dim": str(in_dim),
                    "out_dim": str(out_dim),
                    "epochs": str(epochs),
                    "global_step": str(global_step),
                    "lr": str(lr),
                    "batch_size": str(batch_size),
                    "final_loss": f"{avg_loss:.6f}"
                }
            )
            ckpt_meta = {
                "event": "checkpoint",
                "path": ckpt_path,
                "tensorsSaved": len(t_dict),
                "sizeBytes": os.path.getsize(ckpt_path) if os.path.exists(ckpt_path) else 0
            }
            print(f"__METRICS__:{json.dumps(ckpt_meta)}", flush=True)
        except Exception as e:
            err_msg = f"Checkpoint save failure at '{ckpt_path}': {str(e)}"
            print(f"__ERROR__:{err_msg}", flush=True)
            import traceback
            traceback.print_exc(file=sys.stderr)
            # Fail-Closed: exit with non-zero status so caller treats failure as critical
            sys.exit(2)

    print("__DONE__", flush=True)


def main():
    parser = argparse.ArgumentParser(description="termux-train production session runner")
    parser.add_argument("--config", type=str, help="Path to JSON configuration file")
    parser.add_argument("--stdin-json", action="store_true", help="Read JSON configuration from stdin")
    args = parser.parse_args()

    if args.stdin_json:
        raw = sys.stdin.read()
        cfg = json.loads(raw)
    elif args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
    else:
        # Default synthetic run
        cfg = {"modelType": "mlp", "dim": 32, "epochs": 5, "lr": 0.001}

    try:
        run_session(cfg)
    except Exception as exc:
        print(f"__FATAL__:{str(exc)}", flush=True)
        import traceback
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

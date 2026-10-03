"""
termux_train.data.vision_dataset
================================
On-device Vision VLM Dataset Loader with persistent visual latent caching.
Supports LLaVA / Qwen2-VL style Image-Text QA pairs with zero repeated encoding overhead.
Open-Source under Apache License 2.0.
"""

import os
import glob
import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple

import termux_train as tt
from termux_train import Tensor, get_backend
from termux_train.checkpoint.safetensors import save_safetensors, load_safetensors


def _simple_tokenize(text: str, vocab_size: int = 1000, max_len: int = 64) -> List[int]:
    """Lightweight zero-dependency tokenization mapping UTF-8 byte hashes to token IDs."""
    words = text.strip().split()
    tokens = [1]  # <bos>
    for w in words:
        tok_id = (hashlib.md5(w.encode("utf-8")).digest()[0] % (vocab_size - 4)) + 4
        tokens.append(int(tok_id))
    tokens.append(2)  # <eos>
    if len(tokens) < max_len:
        tokens.extend([0] * (max_len - len(tokens)))  # <pad>
    else:
        tokens = tokens[:max_len]
        tokens[-1] = 2
    return tokens


class VisionQADataset:
    """
    On-device Vision-Language QA Dataset with SafeTensors Visual Latent Caching.
    Reads either:
      1) A manifest JSON/JSONL file with items: {"image": "...", "question": "...", "answer": "..."}
      2) An image directory where images are paired with .json or .txt QA files.
    """

    def __init__(
        self,
        data_source: str,
        image_dir: Optional[str] = None,
        patch_dim: int = 64,
        num_patches: int = 16,
        max_seq_len: int = 64,
        vocab_size: int = 1000,
        cache_dir: Optional[str] = None,
    ):
        self.data_source = data_source
        self.image_dir = image_dir or (data_source if os.path.isdir(data_source) else os.path.dirname(data_source))
        self.patch_dim = patch_dim
        self.num_patches = num_patches
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.cache_dir = cache_dir or os.path.join(self.image_dir, ".cache", "vision_latents")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.samples: List[Dict[str, Any]] = []
        self._load_samples()

    def _load_samples(self) -> None:
        if os.path.isfile(self.data_source):
            if self.data_source.endswith(".jsonl"):
                with open(self.data_source, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.samples.append(json.loads(line))
            elif self.data_source.endswith(".json"):
                with open(self.data_source, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        self.samples.extend(data)
                    elif isinstance(data, dict):
                        self.samples.append(data)
        elif os.path.isdir(self.data_source):
            # Scan directory for images paired with .json or .txt
            valid_exts = {".png", ".jpg", ".jpeg", ".webp", ".ppm"}
            for root, _, files in os.walk(self.data_source):
                for f in sorted(files):
                    ext = os.path.splitext(f)[1].lower()
                    if ext in valid_exts:
                        img_path = os.path.join(root, f)
                        base_stem = os.path.splitext(img_path)[0]
                        json_path = base_stem + ".json"
                        txt_path = base_stem + ".txt"

                        question = "Describe the image."
                        answer = "A high quality visual scene."

                        if os.path.exists(json_path):
                            try:
                                with open(json_path, "r", encoding="utf-8") as jf:
                                    jdata = json.load(jf)
                                    question = jdata.get("question", question)
                                    answer = jdata.get("answer", answer)
                            except Exception:
                                pass
                        elif os.path.exists(txt_path):
                            try:
                                with open(txt_path, "r", encoding="utf-8") as tf:
                                    answer = tf.read().strip()
                            except Exception:
                                pass

                        self.samples.append({
                            "image": img_path,
                            "question": question,
                            "answer": answer,
                        })

        if not self.samples:
            raise FileNotFoundError(f"No valid Vision-QA samples found in '{self.data_source}'")

    def __len__(self) -> int:
        return len(self.samples)

    def _get_image_cache_path(self, img_path: str) -> str:
        h = hashlib.sha256()
        h.update(os.path.abspath(img_path).encode("utf-8"))
        h.update(str(self.patch_dim).encode("utf-8"))
        h.update(str(self.num_patches).encode("utf-8"))
        if os.path.exists(img_path):
            h.update(str(os.path.getmtime(img_path)).encode("utf-8"))
        return os.path.join(self.cache_dir, f"{h.hexdigest()}.safetensors")

    def _extract_visual_features(self, img_path: str) -> Tensor:
        cache_file = self._get_image_cache_path(img_path)
        if os.path.exists(cache_file):
            try:
                tensors = load_safetensors(cache_file)
                if "visual_features" in tensors:
                    return tensors["visual_features"]
            except Exception:
                pass

        # Extract features (deterministic simulation based on image hash/data)
        b = get_backend()
        img_bytes = b""
        if os.path.exists(img_path):
            with open(img_path, "rb") as f:
                img_bytes = f.read(4096)
        
        seed_hash = hashlib.sha256(img_bytes or img_path.encode("utf-8")).digest()
        flat_feats = []
        for i in range(self.num_patches * self.patch_dim):
            byte_val = seed_hash[i % len(seed_hash)]
            val = (byte_val / 127.5) - 1.0
            flat_feats.append(val)

        feat_data = b.from_data(b.reshape(flat_feats, (self.num_patches, self.patch_dim)), dtype="float32")
        feat_tensor = Tensor(feat_data, dtype="float32", backend=b)

        # Cache persistently
        try:
            save_safetensors({"visual_features": feat_tensor}, cache_file)
        except Exception:
            pass

        return feat_tensor

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        sample = self.samples[idx]
        img_path = sample.get("image", "")
        if not os.path.isabs(img_path):
            img_path = os.path.join(self.image_dir, img_path)

        visual_feat = self._extract_visual_features(img_path)
        
        # Tokenize prompt & answer
        conversations = sample.get("conversations")
        if conversations and isinstance(conversations, list):
            q_text = ""
            a_text = ""
            for c in conversations:
                if c.get("from") in ("human", "user"):
                    q_text = c.get("value", "")
                elif c.get("from") in ("gpt", "assistant"):
                    a_text = c.get("value", "")
        else:
            q_text = sample.get("question", "Describe this image.")
            a_text = sample.get("answer", "A high quality picture.")

        full_text = f"USER: {q_text} ASSISTANT: {a_text}"
        tokens = _simple_tokenize(full_text, vocab_size=self.vocab_size, max_len=self.max_seq_len)

        # Input is tokens[:-1], Target is tokens[1:] for autoregressive LM
        b = get_backend()
        input_ids = Tensor(tokens[:-1], dtype="int64", backend=b)
        target_ids = Tensor(tokens[1:], dtype="int64", backend=b)

        return visual_feat, input_ids, target_ids

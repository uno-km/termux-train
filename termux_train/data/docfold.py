"""
termux_train.data.docfold
=========================
DocFold Dataset Pipeline for Structuring and Folding Sequences on Mobile.
Maps unstructured text to structured token sequences (<DOC>, <HEADER>, <VALUE>, <END>).
"""

import json
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Iterator
from ..tensor import Tensor
from ..tokenization import WordTokenizer, PAD_ID


DOC_START_TOKEN = "<DOC>"
HEADER_START_TOKEN = "<HEADER>"
VALUE_START_TOKEN = "<VALUE>"
DOC_END_TOKEN = "<END>"

DOCFOLD_SPECIAL_TOKENS = [
    DOC_START_TOKEN,
    HEADER_START_TOKEN,
    VALUE_START_TOKEN,
    DOC_END_TOKEN,
]


@dataclass
class DocFoldRecord:
    raw_text: str
    header: str
    values: List[str]

    def to_symbolic_sequence(self) -> str:
        val_str = " ".join([f"{VALUE_START_TOKEN} {v}" for v in self.values])
        return f"{DOC_START_TOKEN} {HEADER_START_TOKEN} {self.header} {val_str} {DOC_END_TOKEN}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "header": self.header,
            "values": self.values
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DocFoldRecord':
        return cls(
            raw_text=data["raw_text"],
            header=data["header"],
            values=list(data["values"])
        )


class DocFoldDataset:
    """
    Dataset container for DocFold sequence mapping tasks.
    Supports in-memory lists, JSONL disk serialization, and tokenized tensor batch generation.
    """

    def __init__(
        self,
        records: Optional[List[DocFoldRecord]] = None,
        tokenizer: Optional[WordTokenizer] = None
    ):
        self.records: List[DocFoldRecord] = records or []
        self.tokenizer = tokenizer or WordTokenizer()
        if not self.tokenizer.is_built:
            self._fit_tokenizer()

    def _fit_tokenizer(self) -> None:
        all_texts = []
        for r in self.records:
            all_texts.append(r.raw_text)
            all_texts.append(r.to_symbolic_sequence())

        if all_texts:
            self.tokenizer.build_vocab(all_texts)

    def add_record(self, record: DocFoldRecord) -> None:
        self.records.append(record)
        self._fit_tokenizer()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> DocFoldRecord:
        return self.records[idx]

    def save_jsonl(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            for r in self.records:
                f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load_jsonl(cls, filepath: str) -> 'DocFoldDataset':
        records = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    records.append(DocFoldRecord.from_dict(data))
        return cls(records=records)

    @classmethod
    def create_toy_dataset(cls) -> 'DocFoldDataset':
        toy_records = [
            DocFoldRecord(
                raw_text="Invoice #1042 Total: $450 Vendor: Acme Corp",
                header="Invoice",
                values=["1042", "$450", "Acme Corp"]
            ),
            DocFoldRecord(
                raw_text="Receipt #881 Total: $25 Vendor: Metro Cafe",
                header="Receipt",
                values=["881", "$25", "Metro Cafe"]
            ),
            DocFoldRecord(
                raw_text="Report #309 Total: $1200 Vendor: Global Tech",
                header="Report",
                values=["309", "$1200", "Global Tech"]
            ),
            DocFoldRecord(
                raw_text="Order #774 Total: $89 Vendor: Fast Supply",
                header="Order",
                values=["774", "$89", "Fast Supply"]
            ),
        ]
        return cls(records=toy_records)

    def create_batches(
        self,
        batch_size: int = 2,
        max_seq_len: int = 32,
        ignore_index: int = -100
    ) -> List[Tuple[Tensor, Tensor]]:
        """
        Tokenize and batch sequences for sequence-to-sequence / autoregressive training.
        Returns list of (input_tensor, target_tensor) tuples.
        """
        batches = []
        pad_id = PAD_ID

        # Each sample is: <BOS> raw_text <SEP> symbolic_sequence <EOS>
        all_samples = []
        for r in self.records:
            seq_text = f"{r.raw_text} -> {r.to_symbolic_sequence()}"
            tokens = self.tokenizer.encode(seq_text, add_bos=True, add_eos=True)
            if len(tokens) > max_seq_len:
                tokens = tokens[:max_seq_len]
            all_samples.append(tokens)

        for i in range(0, len(all_samples), batch_size):
            b_samples = all_samples[i:i + batch_size]
            b_inps = []
            b_tgts = []

            for tokens in b_samples:
                # Pad to max_seq_len
                cur_len = len(tokens)
                if cur_len < 2:
                    continue
                inp_seq = tokens[:-1]
                tgt_seq = tokens[1:]

                pad_count = (max_seq_len - 1) - len(inp_seq)
                if pad_count > 0:
                    inp_seq = inp_seq + [pad_id] * pad_count
                    tgt_seq = tgt_seq + [ignore_index] * pad_count
                else:
                    inp_seq = inp_seq[:max_seq_len - 1]
                    tgt_seq = tgt_seq[:max_seq_len - 1]

                b_inps.append(inp_seq)
                b_tgts.append(tgt_seq)

            if b_inps:
                x_t = Tensor(b_inps, dtype="int64")
                y_t = Tensor(b_tgts, dtype="int64")
                batches.append((x_t, y_t))

        return batches

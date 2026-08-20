"""
tests/test_docfold.py
=====================
Test Suite for DocFold Dataset Pipeline:
  - Record Serialization & Symbolic Sequence Mapping
  - JSONL Save and Load
  - Tokenization, Batch Generation, and Padding Invariants
"""

import os
import tempfile
import pytest
from termux_train.data import DocFoldRecord, DocFoldDataset, DOC_START_TOKEN, HEADER_START_TOKEN, VALUE_START_TOKEN, DOC_END_TOKEN


def test_docfold_record_serialization():
    rec = DocFoldRecord(
        raw_text="Receipt #42 Total: $10 Vendor: Coffee Shop",
        header="Receipt",
        values=["42", "$10", "Coffee Shop"]
    )
    sym_seq = rec.to_symbolic_sequence()
    assert DOC_START_TOKEN in sym_seq
    assert HEADER_START_TOKEN in sym_seq
    assert VALUE_START_TOKEN in sym_seq
    assert DOC_END_TOKEN in sym_seq
    assert "Receipt" in sym_seq
    assert "Coffee Shop" in sym_seq

    d = rec.to_dict()
    assert d["header"] == "Receipt"
    rec2 = DocFoldRecord.from_dict(d)
    assert rec2.raw_text == rec.raw_text
    assert rec2.values == rec.values


def test_docfold_dataset_jsonl_roundtrip():
    ds1 = DocFoldDataset.create_toy_dataset()
    assert len(ds1) == 4

    with tempfile.TemporaryDirectory() as tmpdir:
        jsonl_path = os.path.join(tmpdir, "docfold.jsonl")
        ds1.save_jsonl(jsonl_path)
        assert os.path.exists(jsonl_path)

        ds2 = DocFoldDataset.load_jsonl(jsonl_path)
        assert len(ds2) == 4
        assert ds2[0].header == "Invoice"
        assert ds2[1].header == "Receipt"


def test_docfold_batch_generation_and_padding():
    ds = DocFoldDataset.create_toy_dataset()
    batches = ds.create_batches(batch_size=2, max_seq_len=24, ignore_index=-100)

    assert len(batches) == 2
    for x, y in batches:
        assert x.shape == (2, 23)
        assert y.shape == (2, 23)
        assert x.dtype == "int64"
        assert y.dtype == "int64"

"""
tests/test_package_identity.py
==============================
Verify official package import contract and ensure legacy aliases are completely unsupported.
"""

import importlib
import importlib.util
import termux_train

def test_official_package_import():
    module = importlib.import_module("termux_train")
    assert module is termux_train
    assert hasattr(module, "Tensor")
    assert hasattr(module, "nn")
    assert hasattr(module, "__version__")

def test_legacy_package_is_removed():
    assert importlib.util.find_spec("termux_torch") is None

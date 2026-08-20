"""
termux_train.backend
====================
Pluggable backend switcher and manager.
"""

from typing import Dict, List, Optional
from .base import BaseBackend
from .python_backend import PythonBackend

_BACKENDS: Dict[str, BaseBackend] = {
    "python": PythonBackend()
}

try:
    from .numpy_backend import NumPyBackend, NUMPY_AVAILABLE
    if NUMPY_AVAILABLE:
        _BACKENDS["numpy"] = NumPyBackend()
except ImportError:
    pass

# Default backend selection: Prefer numpy if installed, fallback to pure python
_CURRENT_BACKEND_NAME: str = "numpy" if "numpy" in _BACKENDS else "python"

def available_backends() -> List[str]:
    """Return list of available compute backend names."""
    return list(_BACKENDS.keys())

def get_backend(name: Optional[str] = None) -> BaseBackend:
    """Return the specified or currently active compute backend."""
    if name is not None:
        name_clean = name.lower().strip()
        if name_clean not in _BACKENDS:
            raise ValueError(f"Backend '{name}' is not available. Available: {available_backends()}")
        return _BACKENDS[name_clean]
    return _BACKENDS[_CURRENT_BACKEND_NAME]

def set_backend(name: str = "auto") -> BaseBackend:
    """Switch the active compute backend ('auto', 'python', 'numpy')."""
    global _CURRENT_BACKEND_NAME
    name_clean = name.lower().strip()
    if name_clean == "auto":
        _CURRENT_BACKEND_NAME = "numpy" if "numpy" in _BACKENDS else "python"
        return _BACKENDS[_CURRENT_BACKEND_NAME]
    if name_clean not in _BACKENDS:
        raise ValueError(f"Backend '{name}' is not available. Available: {available_backends()} or 'auto'")
    _CURRENT_BACKEND_NAME = name_clean
    return _BACKENDS[_CURRENT_BACKEND_NAME]

__all__ = [
    "BaseBackend",
    "PythonBackend",
    "get_backend",
    "set_backend",
    "available_backends",
]

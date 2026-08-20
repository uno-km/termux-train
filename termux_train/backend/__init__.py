"""
termux_train.backend
====================
Pluggable, thread-safe backend switcher and manager using contextvars.
"""

from contextvars import ContextVar
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
_DEFAULT_BACKEND: str = "numpy" if "numpy" in _BACKENDS else "python"
_ACTIVE_BACKEND_VAR: ContextVar[str] = ContextVar("termux_train_active_backend", default=_DEFAULT_BACKEND)

def available_backends() -> List[str]:
    """Return list of available compute backend names."""
    return list(_BACKENDS.keys())

def get_backend(name: Optional[str] = None) -> BaseBackend:
    """Return the specified or currently active compute backend (thread-safe)."""
    if name is not None:
        name_clean = name.lower().strip()
        if name_clean not in _BACKENDS:
            raise ValueError(f"Backend '{name}' is not available. Available: {available_backends()}")
        return _BACKENDS[name_clean]
    current = _ACTIVE_BACKEND_VAR.get()
    return _BACKENDS[current]

def set_backend(name: str = "auto") -> BaseBackend:
    """Switch the active compute backend ('auto', 'python', 'numpy') for the current context."""
    name_clean = name.lower().strip()
    if name_clean == "auto":
        target = "numpy" if "numpy" in _BACKENDS else "python"
        _ACTIVE_BACKEND_VAR.set(target)
        return _BACKENDS[target]
    if name_clean not in _BACKENDS:
        raise ValueError(f"Backend '{name}' is not available. Available: {available_backends()} or 'auto'")
    _ACTIVE_BACKEND_VAR.set(name_clean)
    return _BACKENDS[name_clean]

__all__ = [
    "BaseBackend",
    "PythonBackend",
    "get_backend",
    "set_backend",
    "available_backends",
]

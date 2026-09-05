"""
termux_train.backend
====================
Pluggable, thread-safe backend switcher and manager using contextvars.

Backend 우선순위 (set_backend('auto')):
  vulkan > numpy > python

사용 방법:
    from termux_train import set_backend
    set_backend('auto')    # Vulkan 자동 선택 (없으면 numpy)
    set_backend('vulkan')  # Vulkan 강제 (ameva-runtime 필요)
    set_backend('numpy')   # NumPy NEON
    set_backend('python')  # 순수 Python
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
except ImportError as _np_err:
    import logging
    logging.getLogger(__name__).debug("numpy backend unavailable (%s), using pure-python", _np_err)

# [신규] VulkanBackend — ameva-runtime 설치 시 자동 활성화
# pip install termux-train[vulkan]  또는  pip install ameva-runtime
try:
    from .vulkan_backend import VulkanBackend
    _BACKENDS["vulkan"] = VulkanBackend()
except ImportError as _vk_imp_err:
    import logging
    logging.getLogger(__name__).debug("vulkan backend not installed: %s", _vk_imp_err)
except (RuntimeError, OSError) as _vk_init_err:
    import logging
    logging.getLogger(__name__).warning("vulkan backend initialization failed (%s); falling back to CPU", _vk_init_err)

# Default: vulkan > numpy > python
def _select_default() -> str:
    if "vulkan" in _BACKENDS:
        return "vulkan"
    if "numpy" in _BACKENDS:
        return "numpy"
    return "python"

_DEFAULT_BACKEND: str = _select_default()
_ACTIVE_BACKEND_VAR: ContextVar[str] = ContextVar("termux_train_active_backend", default=_DEFAULT_BACKEND)

def available_backends() -> List[str]:
    """Return list of available compute backend names."""
    return list(_BACKENDS.keys())

def get_backend(name: Optional[str] = None) -> BaseBackend:
    """Return the specified or currently active compute backend (thread-safe)."""
    if name is not None:
        name_clean = name.lower().strip()
        if name_clean not in _BACKENDS:
            _msg = f"Backend '{name}' is not available. Available: {available_backends()}"
            if name_clean == "vulkan":
                _msg += "\n[Action] pip install ameva-runtime  또는  pip install termux-train[vulkan]"
            raise ValueError(_msg)
        return _BACKENDS[name_clean]
    current = _ACTIVE_BACKEND_VAR.get()
    return _BACKENDS[current]

def set_backend(name: str = "auto") -> BaseBackend:
    """Switch the active compute backend for the current context.

    Args:
        name: 'auto' (vulkan > numpy > python), 'vulkan', 'numpy', 'python'.

    Returns:
        Selected backend instance.

    Raises:
        ValueError: If requested backend is unavailable (includes install instruction for 'vulkan').
    """
    name_clean = name.lower().strip()
    if name_clean == "auto":
        target = _select_default()
        _ACTIVE_BACKEND_VAR.set(target)
        return _BACKENDS[target]
    if name_clean not in _BACKENDS:
        _msg = f"Backend '{name}' is not available. Available: {available_backends()} or 'auto'"
        if name_clean == "vulkan":
            _msg += "\n[Action] pip install ameva-runtime  또는  pip install termux-train[vulkan]"
        raise ValueError(_msg)
    _ACTIVE_BACKEND_VAR.set(name_clean)
    return _BACKENDS[name_clean]

__all__ = [
    "BaseBackend",
    "PythonBackend",
    "get_backend",
    "set_backend",
    "available_backends",
]

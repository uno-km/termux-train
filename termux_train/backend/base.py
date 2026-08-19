"""
termux_train.backend.base
=========================
Abstract Base Class defining the pluggable tensor compute engine interface.
"""

from abc import ABC, abstractmethod
from typing import Any, Tuple, List, Sequence, Union, Callable, Optional

Shape = Tuple[int, ...]

class BaseBackend(ABC):
    """Abstract Base Class for all termux-train compute backends."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of the backend (e.g. 'python', 'numpy', 'neon', 'opencl')."""
        pass

    @abstractmethod
    def from_data(self, data: Any) -> Any:
        """Convert arbitrary python data (scalar, list, array) into the backend native structure."""
        pass

    @abstractmethod
    def get_shape(self, data: Any) -> Shape:
        """Return the dimension tuple (e.g. (2, 3)) of the native data."""
        pass

    @abstractmethod
    def to_flat_list(self, data: Any) -> List[float]:
        """Convert native data to a flat 1D Python list of floats."""
        pass

    @abstractmethod
    def to_nested_list(self, data: Any) -> Any:
        """Convert native data to nested Python lists or scalar."""
        pass

    @abstractmethod
    def zeros(self, shape: Shape) -> Any:
        """Create a tensor filled with zeros."""
        pass

    @abstractmethod
    def ones(self, shape: Shape) -> Any:
        """Create a tensor filled with ones."""
        pass

    @abstractmethod
    def randn(self, shape: Shape, mean: float = 0.0, std: float = 1.0) -> Any:
        """Create a tensor initialized from a Gaussian normal distribution."""
        pass

    @abstractmethod
    def reshape(self, data: Any, new_shape: Shape) -> Any:
        """Reshape tensor data to a new dimension tuple."""
        pass

    @abstractmethod
    def transpose(self, data: Any, axes: Tuple[int, ...] = None) -> Any:
        """Transpose tensor axes (2D matrix transpose by default)."""
        pass

    @abstractmethod
    def add(self, a: Any, b: Any) -> Any:
        """Elementwise addition a + b (with broadcasting)."""
        pass

    @abstractmethod
    def sub(self, a: Any, b: Any) -> Any:
        """Elementwise subtraction a - b (with broadcasting)."""
        pass

    @abstractmethod
    def mul(self, a: Any, b: Any) -> Any:
        """Elementwise multiplication a * b (with broadcasting)."""
        pass

    @abstractmethod
    def div(self, a: Any, b: Any) -> Any:
        """Elementwise division a / b (with broadcasting)."""
        pass

    @abstractmethod
    def pow(self, a: Any, exp: float) -> Any:
        """Elementwise power a ** exp."""
        pass

    @abstractmethod
    def neg(self, a: Any) -> Any:
        """Elementwise negation -a."""
        pass

    @abstractmethod
    def matmul(self, a: Any, b: Any) -> Any:
        """2D/ND Matrix multiplication a @ b."""
        pass

    @abstractmethod
    def sum(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        """Sum reduction along specified axis or all elements."""
        pass

    @abstractmethod
    def mean(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None, keepdims: bool = False) -> Any:
        """Mean reduction along specified axis or all elements."""
        pass

    @abstractmethod
    def relu(self, data: Any) -> Any:
        """Elementwise ReLU max(0, x)."""
        pass

    @abstractmethod
    def sigmoid(self, data: Any) -> Any:
        """Elementwise Sigmoid 1 / (1 + exp(-x))."""
        pass

    @abstractmethod
    def tanh(self, data: Any) -> Any:
        """Elementwise Tanh."""
        pass

    @abstractmethod
    def unbroadcast(self, grad: Any, target_shape: Shape) -> Any:
        """Collapse broadcasted gradient back to target shape."""
        pass

    @abstractmethod
    def clamp(self, data: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Any:
        """Clamp elements of tensor between [min_val, max_val]."""
        pass

    @abstractmethod
    def log(self, data: Any) -> Any:
        """Natural logarithm ln(x)."""
        pass

    @abstractmethod
    def take(self, data: Any, index: int, axis: int = 0) -> Any:
        """Extract a slice along the specified axis (e.g. batch indexing)."""
        pass

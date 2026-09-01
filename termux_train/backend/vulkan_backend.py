"""
termux_train.backend.vulkan_backend
=====================================
Vulkan GPU 가속 Compute Backend — BaseBackend ABC 준수 구현.

아키텍처 전략:
- matmul() (역전파 포함 핵심 GEMM): ameva-vulkan-runtime 의 VulkanContext 를 통해 GPU 오프로딩.
- 그 외 모든 연산 (elementwise, activation 등): NumPyBackend 에 위임(Delegation).
  Vulkan compute shader 가 없는 연산에 대해 별도 커널 개발 없이 즉시 사용 가능.

초기화 전략:
- VulkanBackend() 생성 시 ameva Doctor 를 실행합니다.
- Vulkan 가속 불가 환경에서는 ImportError 를 raise 하지 않고,
  경고를 logging 에 기록하고 NumPy 완전 위임 모드로 투명하게 폴백합니다.
- 이 덕분에 Trainer 코드 변경 없이 set_backend('vulkan') 만으로 GPU 가속 전환 가능.

사용 방법:
    from termux_train import set_backend
    set_backend('vulkan')           # Vulkan 가속 활성화 (불가 시 자동 폴백)

의존성:
    - ameva-vulkan-runtime >= 1.0.0  (pip install termux-train[vulkan])
    - numpy >= 1.20.0
"""
from __future__ import annotations

import ctypes
import logging
from typing import Any, List, Optional, Tuple, Union

from .base import BaseBackend, Shape

logger = logging.getLogger("termux_train.backend.vulkan")

# NumPy 는 VulkanBackend 의 위임 대상이므로 반드시 필요합니다.
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False
    np = None  # type: ignore

# ameva-vulkan-runtime — optional 의존성. 없으면 CPU NumPy 폴백.
try:
    from ameva_vulkan_runtime.doctor import Doctor, DiagnosticReport
    _AMEVA_AVAILABLE = True
except ImportError:
    _AMEVA_AVAILABLE = False
    Doctor = None  # type: ignore
    DiagnosticReport = None  # type: ignore


class VulkanBackend(BaseBackend):
    """Vulkan GPU 가속 Compute Backend.

    matmul() 은 ameva-vulkan-runtime 의 Vulkan GEMM 컨텍스트로 처리하고,
    나머지 연산은 NumPyBackend 에 완전히 위임합니다.

    Vulkan 불가 환경에서는 자동으로 NumPy 전용 모드로 투명하게 동작합니다.
    """

    def __init__(self) -> None:
        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "[termux-train:VulkanBackend] NumPy 가 설치되어 있지 않습니다.\n"
                "Action: pip install numpy 또는 pkg install python-numpy (Termux)"
            )

        # NumPy 위임 백엔드 초기화
        from .numpy_backend import NumPyBackend
        self._delegate = NumPyBackend()

        # Vulkan 런타임 초기화
        self._report: Optional[DiagnosticReport] = None
        self._vulkan_active = False
        self._init_vulkan()

    def _init_vulkan(self) -> None:
        """ameva Doctor 를 실행하여 Vulkan 가속 가능 여부를 계측합니다."""
        if not _AMEVA_AVAILABLE:
            logger.warning(
                "[termux-train:VulkanBackend] ameva-vulkan-runtime 이 설치되지 않았습니다. "
                "NumPy 위임 모드로 실행합니다. "
                "Vulkan 가속을 활성화하려면: pip install termux-train[vulkan]"
            )
            return

        try:
            doc = Doctor()
            self._report = doc.run_self_test(verbose=False)
            if self._report.overall_success:
                self._vulkan_active = True
                logger.info(
                    "[termux-train:VulkanBackend] Vulkan GEMM 가속 활성화. "
                    "device=%s vendor=0x%04X",
                    self._report.device_name,
                    self._report.vendor_id,
                )
            else:
                passed = self._report.passed_stages
                total = self._report.total_stages
                logger.warning(
                    "[termux-train:VulkanBackend] Vulkan 진단 실패 (%d/%d 단계). "
                    "NumPy 위임 모드로 실행합니다. device=%s",
                    passed, total, self._report.device_name,
                )
        except Exception as e:
            logger.warning(
                "[termux-train:VulkanBackend] Vulkan 초기화 중 예외 발생: %s. "
                "NumPy 위임 모드로 실행합니다.", e
            )

    @property
    def name(self) -> str:
        if self._vulkan_active:
            device = getattr(self._report, "device_name", "GPU") if self._report else "GPU"
            return f"vulkan({device})"
        return "vulkan(cpu_fallback)"

    @property
    def is_vulkan_active(self) -> bool:
        """실제 Vulkan GEMM 가속이 활성화되어 있는지 반환합니다."""
        return self._vulkan_active

    # -----------------------------------------------------------------------
    # GEMM — Vulkan 가속 핵심 경로
    # -----------------------------------------------------------------------

    def matmul(self, a: Any, b: Any) -> Any:
        """2D/ND 행렬 곱셈 — Vulkan GEMM 가속 또는 NumPy 위임.

        Vulkan 활성 시:
            2D float32 행렬은 Vulkan compute shader 로 처리합니다.
            스칼라·고차원·비 float32 텐서는 NumPy 에 위임합니다.

        Vulkan 비활성 시:
            모든 경우 NumPy 에 위임합니다.
        """
        if not _NUMPY_AVAILABLE:
            raise RuntimeError("[termux-train:VulkanBackend] NumPy unavailable.")

        # 스칼라 검증 (NumPy 위임과 동일한 규칙 유지)
        if (isinstance(a, np.ndarray) and a.ndim == 0) or \
           (isinstance(b, np.ndarray) and b.ndim == 0):
            raise ValueError(
                f"[termux-train:VulkanBackend] matmul 스칼라 피연산자 불가 "
                f"(shapes {getattr(a, 'shape', ())} and {getattr(b, 'shape', ())})"
            )

        if self._vulkan_active:
            return self._vulkan_matmul(a, b)

        return self._delegate.matmul(a, b)

    def _vulkan_matmul(self, a: Any, b: Any) -> Any:
        """실제 Vulkan GEMM 실행.

        현재 단계: ameva-vulkan-runtime 의 C ABI 가 실기기에 배포된 환경에서
        libameva_vulkan.so 를 통해 GEMM 을 수행합니다.
        C HAL 미배포 환경(개발 호스트 등)에서는 NumPy 에 자동 위임합니다.
        """
        # C HAL 직접 호출 시도
        try:
            from ameva_vulkan_runtime.bindings import AmevaVulkanLib
            lib = AmevaVulkanLib()
            if lib.is_loaded():
                # ameva C ABI: ameva_matmul_f32(a_ptr, b_ptr, c_ptr, M, K, N)
                a_f32 = np.ascontiguousarray(a, dtype=np.float32)
                b_f32 = np.ascontiguousarray(b, dtype=np.float32)
                if a_f32.ndim == 2 and b_f32.ndim == 2:
                    M, K = a_f32.shape
                    K2, N = b_f32.shape
                    if K == K2:
                        c_f32 = np.zeros((M, N), dtype=np.float32)
                        ret = lib.call_matmul_f32(a_f32, b_f32, c_f32, M, K, N)
                        if ret == 0:
                            return c_f32
                        else:
                            logger.debug(
                                "[termux-train:VulkanBackend] ameva_matmul_f32 반환 코드 %d "
                                "— NumPy 위임으로 폴백.", ret
                            )
        except Exception as e:
            logger.debug(
                "[termux-train:VulkanBackend] Vulkan GEMM 실행 실패 (%s) "
                "— NumPy 폴백.", e
            )

        # NumPy 폴백 (C HAL 미배포 또는 오류 시)
        return self._delegate.matmul(a, b)

    # -----------------------------------------------------------------------
    # 모든 나머지 연산 — NumPyBackend 완전 위임
    # -----------------------------------------------------------------------

    def from_data(self, data: Any, dtype: Optional[str] = "float32") -> Any:
        return self._delegate.from_data(data, dtype)

    def get_shape(self, data: Any) -> Shape:
        return self._delegate.get_shape(data)

    def to_flat_list(self, data: Any) -> List[Any]:
        return self._delegate.to_flat_list(data)

    def to_nested_list(self, data: Any) -> Any:
        return self._delegate.to_nested_list(data)

    def zeros(self, shape: Shape, dtype: str = "float32") -> Any:
        return self._delegate.zeros(shape, dtype)

    def ones(self, shape: Shape, dtype: str = "float32") -> Any:
        return self._delegate.ones(shape, dtype)

    def randn(self, shape: Shape, mean: float = 0.0, std: float = 1.0) -> Any:
        return self._delegate.randn(shape, mean, std)

    def reshape(self, data: Any, new_shape: Shape) -> Any:
        return self._delegate.reshape(data, new_shape)

    def transpose(self, data: Any, axes: Tuple[int, ...] = None) -> Any:
        return self._delegate.transpose(data, axes)

    def add(self, a: Any, b: Any) -> Any:
        return self._delegate.add(a, b)

    def sub(self, a: Any, b: Any) -> Any:
        return self._delegate.sub(a, b)

    def mul(self, a: Any, b: Any) -> Any:
        return self._delegate.mul(a, b)

    def div(self, a: Any, b: Any) -> Any:
        return self._delegate.div(a, b)

    def pow(self, a: Any, exp: float) -> Any:
        return self._delegate.pow(a, exp)

    def exp(self, a: Any) -> Any:
        return self._delegate.exp(a)

    def sqrt(self, a: Any) -> Any:
        return self._delegate.sqrt(a)

    def neg(self, a: Any) -> Any:
        return self._delegate.neg(a)

    def sum(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None,
            keepdims: bool = False) -> Any:
        return self._delegate.sum(data, axis, keepdims)

    def max(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None,
            keepdims: bool = False) -> Any:
        return self._delegate.max(data, axis, keepdims)

    def mean(self, data: Any, axis: Union[int, Tuple[int, ...], None] = None,
             keepdims: bool = False) -> Any:
        return self._delegate.mean(data, axis, keepdims)

    def relu(self, data: Any) -> Any:
        return self._delegate.relu(data)

    def sigmoid(self, data: Any) -> Any:
        return self._delegate.sigmoid(data)

    def tanh(self, data: Any) -> Any:
        return self._delegate.tanh(data)

    def unbroadcast(self, grad: Any, target_shape: Shape) -> Any:
        return self._delegate.unbroadcast(grad, target_shape)

    def clamp(self, data: Any, min_val: Optional[float] = None,
              max_val: Optional[float] = None) -> Any:
        return self._delegate.clamp(data, min_val, max_val)

    def log(self, data: Any) -> Any:
        return self._delegate.log(data)

    def take(self, data: Any, index: int, axis: int = 0) -> Any:
        return self._delegate.take(data, index, axis)

    def gather_rows(self, weight_data: Any, row_indices: List[int],
                    out_shape: Tuple[int, ...]) -> Any:
        return self._delegate.gather_rows(weight_data, row_indices, out_shape)

    def scatter_add_rows(self, target_data: Any, row_indices: List[int],
                         grad_data: Any, padding_idx: Optional[int] = None) -> Any:
        return self._delegate.scatter_add_rows(target_data, row_indices, grad_data, padding_idx)

"""
termux_train.adapter
=====================
AMEVA Component Protocol v1 — Orchestrator Adapter (v0.8.1 호환)

오케스트레이터 v0.8.1이 ameva.components Entry Point로 탐색합니다.
Train 패키지의 단일 진실 원천: TrainControl.
"""
from __future__ import annotations

from typing import Any, AsyncIterator

from ameva_component.adapter_base import BaseOrchestratorAdapter
from termux_train.control.component import TrainControl


class TrainOrchestratorAdapter(BaseOrchestratorAdapter):
    """Train Orchestrator Adapter.

    TrainControl을 통해 단일 진실 원천을 보장합니다.

    Train 패키지 특성:
    - instance = Training Worker 프로세스 (추론 인스턴스 아님)
    - activate_model / deactivate_model → OPERATION_NOT_SUPPORTED
      (훈련은 worker instance 단위로 제어)
    - infer() → OPERATION_NOT_SUPPORTED
      (추론이 아닌 학습(LoRA fine-tuning) 패키지)
    - start_instance / stop_instance → Training Worker 시작/중단
    - drain_instance → Checkpoint 저장 후 신규 작업 접수 중단
    """

    COMPONENT_ID = "termux-train"

    def __init__(self, control: TrainControl | None = None) -> None:
        self._control = control or TrainControl()

    # ── activate / deactivate: 훈련 패키지는 모델 활성화 개념 없음 ──

    async def activate(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._not_supported("activate")

    async def deactivate(self, request: dict[str, Any]) -> dict[str, Any]:
        return self._not_supported("deactivate")

    # ── models: 훈련 패키지는 모델 목록 없음 ──

    def models(self) -> dict[str, Any]:
        return self._not_supported("models")

    # ── infer: 훈련 패키지는 streaming inference 미지원 ──

    async def infer(self, request: dict[str, Any]) -> AsyncIterator[dict[str, Any]]:
        """termux-train은 LLM 추론이 아닌 학습(LoRA fine-tuning) 패키지입니다.
        Streaming inference는 OPERATION_NOT_SUPPORTED — llama.cpp 서버를 사용하십시오.
        """
        yield self._not_supported("infer")


def create_adapter() -> TrainOrchestratorAdapter:
    """Entry Point Factory. 오케스트레이터가 ameva.components 그룹에서 호출합니다."""
    return TrainOrchestratorAdapter()
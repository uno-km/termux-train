"""
termux_train.control.status — Train 상태 파일 Writer + Heartbeat

훈련 특화 동작:
  - notify_job_start() → Training Run 시작 (active_training_runs++)
  - notify_job_end()   → Training Run 완료/중단
  - notify_error()     → 훈련 실패 기록
"""
from __future__ import annotations
from typing import Any
from ameva_component.heartbeat import HeartbeatWriter


class TrainStatusWriter(HeartbeatWriter):
    """TrainControl 상태를 10초마다 상태 파일에 원자적으로 기록합니다."""

    def __init__(self, control: Any) -> None:
        super().__init__(control, name="train")

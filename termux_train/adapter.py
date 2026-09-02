"""termux_train.adapter — Orchestrator Adapter."""
from __future__ import annotations
from termux_train.control.component import TrainControl

class TrainOrchestratorAdapter:
    def __init__(self, control: TrainControl | None = None) -> None:
        self._control = control or TrainControl()
    def info(self) -> dict: return self._control.component_info()
    def health(self) -> dict: return self._control.doctor_lite()
    def models(self) -> dict: return self._control.list_models()
    def instances(self) -> dict: return self._control.list_instances()
    async def start_worker(self, req: dict) -> dict: return await self._control.start_instance(req)
    async def drain_worker(self, instance_id: str) -> dict: return await self._control.drain_instance(instance_id)
    async def stop_worker(self, instance_id: str) -> dict: return await self._control.stop_instance(instance_id)

"""
termux_train.control.component
AMEVA Component Protocol v1 — TrainControl

instance = Training Worker (추론 인스턴스 아님)
active_jobs = Training Run 수 (추론 요청 아님)
drain = Checkpoint 저장 후 중단 또는 새 작업 접수 중단
activate_model → OPERATION_NOT_SUPPORTED (훈련은 인스턴스로 제어)
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from ameva_component import (
    ActivationLock, ComponentInfo, ComponentStateFile,
    ControlMode, InstanceRegistry, InstanceState, InstanceStatus,
    ModelRegistry, ModelState, ModelNotFound,
    OperationNotSupported, now_timestamps, log_stderr, PROTOCOL_COMPONENT,
)
from ameva_component.control import ComponentControl


class TrainControl(ComponentControl):
    """
    termux-train ComponentControl.

    훈련 프레임워크 전용:
    - instance = Training Worker 프로세스
    - active_jobs = 현재 실행 중인 Training Run
    - drain = Checkpoint 저장 시그널
    - activate_model → OPERATION_NOT_SUPPORTED
    """

    COMPONENT_ID   = "termux-train"
    COMPONENT_TYPE = "train"
    CAPABILITIES   = ("train.lora",)

    DEFAULT_CHECKPOINT_DIR = Path.home() / ".termux-train" / "checkpoints"
    DEFAULT_DATASETS_DIR   = Path.home() / ".termux-train" / "datasets"

    def __init__(self) -> None:
        self._state_file = ComponentStateFile(self.COMPONENT_ID)
        self._model_reg  = ModelRegistry(self.COMPONENT_ID)
        self._inst_reg   = InstanceRegistry(self.COMPONENT_ID)
        # Phase 4: Heartbeat Writer
        from termux_train.control.status import TrainStatusWriter
        self._heartbeat = TrainStatusWriter(self)

    def _get_version(self) -> str:
        try:
            from termux_train import __version__; return __version__
        except Exception: return "1.1.0"

    def component_info(self) -> dict:
        info = ComponentInfo(
            protocol=PROTOCOL_COMPONENT, component_id=self.COMPONENT_ID,
            component_type=self.COMPONENT_TYPE, version=self._get_version(),
            capabilities=self.CAPABILITIES,
        )
        info.validate()
        return info.to_dict()

    def doctor_lite(self) -> dict:
        ts = now_timestamps()
        state_data = self._state_file.read()
        stale = self._state_file.is_stale(threshold_ms=30_000)
        pid, pid_alive = self._check_pid()
        instances = self._inst_reg.list_all()
        busy = [i for i in instances if i.state == InstanceState.BUSY]

        # 백엔드 가용 여부만 (실제 훈련 실행 금지)
        backend_info = self._check_backend()
        ready = any(backend_info.values())
        degraded = stale or not any(backend_info.values())

        return {
            "protocol": "ameva-component-status/1",
            "component_id": self.COMPONENT_ID, "component_type": self.COMPONENT_TYPE,
            "version": self._get_version(), "ready": ready, "degraded": degraded,
            **ts,
            "process": {"running": pid_alive, "pid": pid},
            "capabilities": list(self.CAPABILITIES),
            "training_workers": len(instances),
            "active_training_runs": len(busy),
            "backends": backend_info,
            "errors": [state_data.get("last_error")] if state_data and state_data.get("last_error") else [],
            "state_file": {"path": str(self._state_file.path), "stale": stale,
                           "updated_at": state_data.get("updated_at") if state_data else None},
        }

    def _check_pid(self) -> tuple[int | None, bool]:
        """P0-5: 상태 파일 기반 Training Worker PID 활성 여부. 'pid 없음'과 '검사 실패' 구분."""
        import logging
        _log = logging.getLogger(__name__)

        state_data = self._state_file.read()
        if state_data:
            pid = state_data.get("process", {}).get("pid")
            if pid:
                try:
                    os.kill(pid, 0)
                    return pid, True
                except ProcessLookupError:
                    return pid, False
                except PermissionError as perm_err:
                    _log.warning("[train] Worker PID %d PermissionError: %s", pid, perm_err)
                    return pid, False
                except OSError as os_err:
                    _log.warning("[train] Worker PID %d OSError: %s", pid, os_err)
                    return pid, False
        return None, False

    def _check_backend(self) -> dict:
        result = {}
        for name, module in [
            ("python", "termux_train.backend.python_backend"),
            ("numpy",  "termux_train.backend.numpy_backend"),
            ("vulkan", "termux_train.backend.vulkan_backend"),
        ]:
            try:
                __import__(module); result[name] = True
            except ImportError: result[name] = False
        return result

    def doctor_full(self) -> dict:
        lite = self.doctor_lite()
        try:
            import io, contextlib, argparse
            from termux_train.cli import main as train_main
            # doctor 서브커맨드 시도
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                train_main(["doctor"])
            lite["doctor_output"] = buf.getvalue()
        except Exception as e:
            lite["doctor_error"] = str(e)
        lite["doctor_level"] = "full"
        return lite

    def list_models(self) -> dict:
        """Base Model 목록 (훈련 대상으로 등록된 모델)."""
        reg_map = {m["model_id"]: m for m in self._model_reg.list_all()}
        return {"models": list(reg_map.values()), "total": len(reg_map),
                "checkpoint_dir": str(self.DEFAULT_CHECKPOINT_DIR),
                "note": "Models here are training base models, not inference models"}

    def model_status(self, model_id: str | None = None) -> dict:
        if model_id:
            rec = self._model_reg.get(model_id)
            if rec is None: raise ModelNotFound(model_id)
            return {"model": rec}
        return self.list_models()

    def install_model(self, request: dict) -> dict:
        """Base Model 다운로드."""
        from ameva_component import ModelInstaller
        url = request.get("url", ""); filename = request.get("filename", "")
        sha256 = request.get("sha256", ""); expected_bytes = int(request.get("expected_bytes", 0))
        model_id = request.get("model_id") or Path(filename).stem
        dest = self.DEFAULT_CHECKPOINT_DIR
        dest.mkdir(parents=True, exist_ok=True)
        installer = ModelInstaller(self.COMPONENT_ID, dest, self._model_reg)
        return installer.install(url=url, filename=filename, sha256=sha256,
                                 expected_bytes=expected_bytes, model_id=model_id)

    async def activate_model(self, request: dict) -> dict:
        """훈련 프레임워크에서 activate는 지원하지 않습니다. 인스턴스로 훈련을 제어하세요."""
        raise OperationNotSupported("activate_model", self.COMPONENT_ID)

    async def deactivate_model(self, request: dict) -> dict:
        raise OperationNotSupported("deactivate_model", self.COMPONENT_ID)

    def list_instances(self) -> dict:
        """Training Worker 목록."""
        instances = self._inst_reg.list_all()
        return {
            "training_workers": [i.to_dict() for i in instances],
            "total": len(instances),
            "note": "These are Training Workers, not inference instances",
        }

    async def start_instance(self, request: dict) -> dict:
        """Training Worker 시작."""
        config = request.get("config", {})
        model_id = request.get("base_model_id", "unknown")
        instance_id = request.get("instance_id") or f"train-worker-{int(time.time())}"
        inst = InstanceStatus(
            instance_id=instance_id, component_id=self.COMPONENT_ID,
            model_id=model_id, state=InstanceState.CREATED,
            active_jobs=0, queue_depth=0, max_concurrency=1,
            backend=request.get("backend", "python"),
            started_at=time.time(), last_heartbeat=time.time(),
            last_error=None, control_mode=ControlMode.SUBPROCESS,
        )
        self._inst_reg.register(inst)
        self._inst_reg.update_state(instance_id, InstanceState.BUSY)
        self._write_state()
        # Phase 4: Heartbeat 시작 (Training Worker 시작 트리거)
        self._heartbeat.start()
        return {"instance_id": instance_id, "state": InstanceState.BUSY.value,
                "note": "Training Worker started"}

    async def drain_instance(self, instance_id: str) -> dict:
        """Checkpoint 저장 후 새 작업 접수 중단."""
        from ameva_component import InstanceNotFound
        if not self._inst_reg.get(instance_id): raise InstanceNotFound(instance_id)
        self._inst_reg.update_state(instance_id, InstanceState.DRAINING)
        log_stderr(f"[train] Drain signal sent to {instance_id} — checkpoint will be saved before stopping")
        return {"instance_id": instance_id, "state": InstanceState.DRAINING.value,
                "note": "Checkpoint save requested"}

    async def stop_instance(self, instance_id: str) -> dict:
        from ameva_component import InstanceNotFound
        if not self._inst_reg.get(instance_id): raise InstanceNotFound(instance_id)
        self._inst_reg.update_state(instance_id, InstanceState.STOPPED)
        self._inst_reg.remove(instance_id)
        # Phase 4: Heartbeat 중단 (정상 종료 트리거)
        remaining = self._inst_reg.list_all()
        if not remaining:
            self._heartbeat.stop()
        else:
            self._write_state()
        return {"instance_id": instance_id, "state": InstanceState.STOPPED.value}


    def _write_state(self, *, ready: bool | None = None, last_error: str | None = None) -> None:
        ts = now_timestamps()
        instances = self._inst_reg.list_all()
        busy = [i for i in instances if i.state == InstanceState.BUSY]
        _ready = any(self._check_backend().values()) if ready is None else ready
        self._state_file.write({
            "protocol": "ameva-component-status/1", "component_id": self.COMPONENT_ID,
            "component_type": self.COMPONENT_TYPE, "version": self._get_version(),
            "ready": _ready, "degraded": not _ready, **ts,
            "training_workers": len(instances), "active_training_runs": len(busy),
            "last_error": last_error,
        })

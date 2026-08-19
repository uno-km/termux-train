"""
termux_train.runtime.trainer
============================
MobileTrainer: Lightweight, Pure-Engine Training Loop Orchestrator with Periodic Safe Checkpointing.
Supports standard full-model training and dedicated On-Device LoRA adapter fine-tuning.
Executes standard forward-backward-step lifecycle, global step management, explicit user stop requests,
reentrant protection, and exact transactional resume capabilities without OS or hardware sensor coupling.
"""

import copy
import os
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from ..tensor import Tensor
from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    save_lora_checkpoint,
    load_lora_checkpoint,
    _validate_model_for_lora_checkpoint,
    _validate_optimizer_for_lora_checkpoint,
)

_CHECKPOINT_MODE_FULL = "full"
_CHECKPOINT_MODE_LORA = "lora"


class MobileTrainer:
    """
    Orchestrates the training and fine-tuning lifecycle for termux-train models.

    Args:
        model: Neural network module to train.
        optimizer: First-order optimizer (e.g. SGD, Adam, AdamW).
        criterion: Loss module (e.g. MSELoss, BCELoss).
        checkpoint_dir: Directory to store periodic checkpoints.
        checkpoint_every_epochs: Frequency of periodic checkpointing by epoch (positive integer).
        checkpoint_every_steps: Frequency of periodic checkpointing by global step (positive integer).
        lora_only: If True, operates in dedicated On-Device LoRA mode, saving and restoring
                   only adapter parameters and adapter-only optimizer states.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: Module,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every_epochs: Optional[int] = None,
        checkpoint_every_steps: Optional[int] = None,
        lora_only: bool = False,
    ):
        if not isinstance(lora_only, bool):
            raise TypeError("lora_only must be a bool")

        if not isinstance(model, Module):
            raise TypeError(f"model must be a Module instance, got {type(model).__name__}")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be an Optimizer instance, got {type(optimizer).__name__}")
        if not isinstance(criterion, Module):
            raise TypeError(f"criterion must be a Module instance, got {type(criterion).__name__}")

        if checkpoint_every_epochs is not None:
            if (
                isinstance(checkpoint_every_epochs, bool)
                or not isinstance(checkpoint_every_epochs, int)
                or checkpoint_every_epochs <= 0
            ):
                raise ValueError(
                    f"checkpoint_every_epochs must be a positive integer, got {checkpoint_every_epochs!r}"
                )

        if checkpoint_every_steps is not None:
            if (
                isinstance(checkpoint_every_steps, bool)
                or not isinstance(checkpoint_every_steps, int)
                or checkpoint_every_steps <= 0
            ):
                raise ValueError(
                    f"checkpoint_every_steps must be a positive integer, got {checkpoint_every_steps!r}"
                )

        if (checkpoint_every_epochs is not None or checkpoint_every_steps is not None) and (
            checkpoint_dir is None or not isinstance(checkpoint_dir, str) or not checkpoint_dir.strip()
        ):
            raise ValueError("checkpoint_dir must be a non-empty string when periodic checkpointing is enabled")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_epochs = checkpoint_every_epochs
        self.checkpoint_every_steps = checkpoint_every_steps
        self.lora_only: bool = lora_only
        self._checkpoint_mode: str = _CHECKPOINT_MODE_LORA if lora_only else _CHECKPOINT_MODE_FULL

        self.current_epoch: int = 0
        self.global_step: int = 0
        self.history: List[Dict[str, Any]] = []
        self._stop_requested: bool = False
        self._is_fitting: bool = False

        self._validate_configuration("initialization")

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _validate_configuration(self, action: str = "operation") -> None:
        """Preflight validation for model, optimizer, and mode-specific invariants."""
        if not isinstance(self.model, Module):
            raise TypeError(f"model must be a Module instance, got {type(self.model).__name__}")
        if not isinstance(self.optimizer, Optimizer):
            raise TypeError(f"optimizer must be an Optimizer instance, got {type(self.optimizer).__name__}")
        if not isinstance(self.criterion, Module):
            raise TypeError(f"criterion must be a Module instance, got {type(self.criterion).__name__}")

        if self._checkpoint_mode == _CHECKPOINT_MODE_LORA:
            _validate_model_for_lora_checkpoint(self.model, action)
            _validate_optimizer_for_lora_checkpoint(self.model, self.optimizer, action)

    def request_stop(self) -> None:
        """Requests the training loop to stop gracefully at the next step boundary."""
        self._stop_requested = True

    def _save_checkpoint_for_mode(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Centralized internal checkpoint saver for full and LoRA modes."""
        self._validate_configuration("saving")

        user_extra = copy.deepcopy(extra) if extra is not None else {}
        if not isinstance(user_extra, dict):
            raise TypeError("extra must be a dict or None")

        combined_extra = copy.deepcopy(user_extra)
        combined_extra["_trainer_history"] = copy.deepcopy(self.history)
        combined_extra["_checkpoint_mode"] = self._checkpoint_mode

        if self._checkpoint_mode == _CHECKPOINT_MODE_LORA:
            save_lora_checkpoint(
                path=path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                global_step=self.global_step,
                extra=combined_extra,
            )
        else:
            save_checkpoint(
                path=path,
                model=self.model,
                optimizer=self.optimizer,
                epoch=self.current_epoch,
                global_step=self.global_step,
                extra=combined_extra,
            )

    def _load_checkpoint_for_mode(self, path: str) -> Dict[str, Any]:
        """Centralized internal checkpoint loader with trainer-level transactional protection."""
        self._validate_configuration("resuming")

        orig_epoch = self.current_epoch
        orig_step = self.global_step
        orig_history = copy.deepcopy(self.history)
        orig_stop = self._stop_requested

        try:
            if self._checkpoint_mode == _CHECKPOINT_MODE_LORA:
                meta = load_lora_checkpoint(
                    path=path,
                    model=self.model,
                    optimizer=self.optimizer,
                )
            else:
                meta = load_checkpoint(
                    path=path,
                    model=self.model,
                    optimizer=self.optimizer,
                )

            self.current_epoch = meta["epoch"]
            self.global_step = meta["global_step"]

            extra = meta.get("extra", {})
            if isinstance(extra, dict) and "_trainer_history" in extra and isinstance(extra["_trainer_history"], list):
                self.history = copy.deepcopy(extra["_trainer_history"])
            else:
                self.history = []

            self._stop_requested = False
            return meta

        except Exception as e:
            self.current_epoch = orig_epoch
            self.global_step = orig_step
            self.history = orig_history
            self._stop_requested = orig_stop
            raise e

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Atomically saves the current training state to path."""
        self._save_checkpoint_for_mode(path=path, extra=extra)

    def resume(self, path: str) -> Dict[str, Any]:
        """Atomically restores model, optimizer, epoch, global_step, and history from checkpoint."""
        return self._load_checkpoint_for_mode(path=path)

    def fit(
        self,
        dataset: Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]],
        epochs: int = 1,
        resume_from: Optional[str] = None,
        on_step_end: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the training / fine-tuning loop.

        Args:
            dataset: Either a list of (x, target) batch pairs, or a single (x, target) tuple.
            epochs: Number of additional epochs to train (positive integer).
            resume_from: Optional path to a checkpoint file to resume state before training.
            on_step_end: Optional callback called after each step with step metrics.
            on_epoch_end: Optional callback called after each epoch with epoch metrics.

        Returns:
            Dict containing training history and final status:
              {"epochs_completed": int, "global_step": int, "stopped_early": bool, "history": list}
        """
        if self._is_fitting:
            raise RuntimeError("Reentrant fit() calls on the same MobileTrainer instance are not permitted")

        if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs <= 0:
            raise ValueError(f"epochs must be a positive integer, got {epochs!r}")

        self._is_fitting = True
        try:
            if resume_from:
                self.resume(resume_from)

            self._validate_configuration("fitting")

            if (
                isinstance(dataset, tuple)
                and len(dataset) == 2
                and isinstance(dataset[0], Tensor)
                and isinstance(dataset[1], Tensor)
            ):
                batches = [(dataset[0], dataset[1])]
            elif (
                isinstance(dataset, list)
                and len(dataset) > 0
                and all(
                    isinstance(b, tuple)
                    and len(b) == 2
                    and isinstance(b[0], Tensor)
                    and isinstance(b[1], Tensor)
                    for b in dataset
                )
            ):
                batches = dataset
            else:
                raise TypeError(
                    "dataset must be a non-empty list of (Tensor, Tensor) batch tuples or a single (Tensor, Tensor) tuple"
                )

            self._stop_requested = False
            start_epoch = self.current_epoch + 1
            end_epoch = self.current_epoch + epochs

            for epoch in range(start_epoch, end_epoch + 1):
                if self._stop_requested:
                    break

                epoch_loss_sum = 0.0
                epoch_steps = 0

                for batch_idx, (x_batch, target_batch) in enumerate(batches):
                    if self._stop_requested:
                        break

                    self.optimizer.zero_grad()
                    prediction = self.model(x_batch)
                    loss = self.criterion(prediction, target_batch)
                    loss.backward()
                    self.optimizer.step()

                    self.global_step += 1
                    step_loss = loss.item()
                    epoch_loss_sum += step_loss
                    epoch_steps += 1

                    step_info = {
                        "epoch": epoch,
                        "batch_idx": batch_idx,
                        "global_step": self.global_step,
                        "loss": step_loss,
                    }

                    if on_step_end:
                        on_step_end(step_info)

                    if (
                        self.checkpoint_dir
                        and self.checkpoint_every_steps
                        and self.global_step % self.checkpoint_every_steps == 0
                    ):
                        ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.global_step}.json")
                        self.save(ckpt_path, extra={"trigger": "step", "loss": step_loss})

                self.current_epoch = epoch
                avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
                epoch_info = {
                    "epoch": epoch,
                    "global_step": self.global_step,
                    "loss": avg_epoch_loss,
                }
                self.history.append(epoch_info)

                if on_epoch_end:
                    on_epoch_end(epoch_info)

                if (
                    self.checkpoint_dir
                    and self.checkpoint_every_epochs
                    and epoch % self.checkpoint_every_epochs == 0
                ):
                    ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.json")
                    self.save(ckpt_path, extra={"trigger": "epoch", "loss": avg_epoch_loss})

            if self.checkpoint_dir:
                latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.json")
                self.save(
                    latest_path,
                    extra={"trigger": "latest", "loss": self.history[-1]["loss"] if self.history else 0.0},
                )

            return {
                "epochs_completed": self.current_epoch,
                "global_step": self.global_step,
                "stopped_early": self._stop_requested,
                "history": copy.deepcopy(self.history),
            }
        finally:
            self._is_fitting = False

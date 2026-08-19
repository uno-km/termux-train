"""
termux_train.runtime.trainer
============================
MobileTrainer: Lightweight, Pure-Engine Training Loop Orchestrator with Periodic Safe Checkpointing.
Executes standard forward-backward-step lifecycle, global step management, explicit user stop requests,
and exact resume capabilities without OS or hardware sensor coupling.
"""

import os
from typing import Optional, Dict, Any, Callable, List, Tuple, Union
from ..nn.module import Module
from ..optim.optimizer import Optimizer
from ..tensor import Tensor
from .checkpoint import save_checkpoint, load_checkpoint

class MobileTrainer:
    """
    Orchestrates the training lifecycle for termux-train models.

    Args:
        model: Neural network module to train.
        optimizer: First-order optimizer (e.g. SGD, Adam, AdamW).
        criterion: Loss module (e.g. MSELoss, BCELoss).
        checkpoint_dir: Directory to store periodic checkpoints.
        checkpoint_every_epochs: Frequency of periodic checkpointing by epoch.
        checkpoint_every_steps: Frequency of periodic checkpointing by global step.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        criterion: Module,
        checkpoint_dir: Optional[str] = None,
        checkpoint_every_epochs: Optional[int] = None,
        checkpoint_every_steps: Optional[int] = None,
    ):
        if not isinstance(model, Module):
            raise TypeError(f"model must be a Module instance, got {type(model).__name__}")
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be an Optimizer instance, got {type(optimizer).__name__}")
        if not isinstance(criterion, Module):
            raise TypeError(f"criterion must be a Module instance, got {type(criterion).__name__}")

        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_every_epochs = checkpoint_every_epochs
        self.checkpoint_every_steps = checkpoint_every_steps

        self.current_epoch: int = 0
        self.global_step: int = 0
        self._stop_requested: bool = False

        if self.checkpoint_dir:
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def request_stop(self) -> None:
        """Requests the training loop to stop gracefully at the next step boundary."""
        self._stop_requested = True

    def save(self, path: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Atomically saves the current training state to path."""
        save_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
            epoch=self.current_epoch,
            global_step=self.global_step,
            extra=extra,
        )

    def resume(self, path: str) -> Dict[str, Any]:
        """Atomically restores model, optimizer, epoch, and global_step from checkpoint."""
        meta = load_checkpoint(
            path=path,
            model=self.model,
            optimizer=self.optimizer,
        )
        self.current_epoch = meta["epoch"]
        self.global_step = meta["global_step"]
        return meta

    def fit(
        self,
        dataset: Union[List[Tuple[Tensor, Tensor]], Tuple[Tensor, Tensor]],
        epochs: int = 1,
        resume_from: Optional[str] = None,
        on_step_end: Optional[Callable[[Dict[str, Any]], None]] = None,
        on_epoch_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> Dict[str, Any]:
        """
        Executes the training loop.

        Args:
            dataset: Either a list of (x, target) batch pairs, or a single (x, target) tuple.
            epochs: Number of epochs to train.
            resume_from: Optional path to a checkpoint file to resume state before training.
            on_step_end: Optional callback called after each step with step metrics.
            on_epoch_end: Optional callback called after each epoch with epoch metrics.

        Returns:
            Dict containing training history and final status:
              {"epochs_completed": int, "global_step": int, "stopped_early": bool, "history": list}
        """
        if resume_from:
            self.resume(resume_from)

        if isinstance(dataset, tuple) and len(dataset) == 2 and isinstance(dataset[0], Tensor):
            batches = [(dataset[0], dataset[1])]
        elif isinstance(dataset, list):
            batches = dataset
        else:
            raise TypeError("dataset must be a list of (x, target) batches or a single (x, target) tuple")

        self._stop_requested = False
        history = []
        start_epoch = self.current_epoch + 1
        end_epoch = self.current_epoch + epochs

        for epoch in range(start_epoch, end_epoch + 1):
            if self._stop_requested:
                break

            self.current_epoch = epoch
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

                # Periodic checkpoint by global step
                if (
                    self.checkpoint_dir
                    and self.checkpoint_every_steps
                    and self.global_step % self.checkpoint_every_steps == 0
                ):
                    ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_step_{self.global_step}.json")
                    self.save(ckpt_path, extra={"trigger": "step", "loss": step_loss})

            avg_epoch_loss = epoch_loss_sum / max(1, epoch_steps)
            epoch_info = {
                "epoch": epoch,
                "global_step": self.global_step,
                "loss": avg_epoch_loss,
            }
            history.append(epoch_info)

            if on_epoch_end:
                on_epoch_end(epoch_info)

            # Periodic checkpoint by epoch
            if (
                self.checkpoint_dir
                and self.checkpoint_every_epochs
                and epoch % self.checkpoint_every_epochs == 0
            ):
                ckpt_path = os.path.join(self.checkpoint_dir, f"checkpoint_epoch_{epoch}.json")
                self.save(ckpt_path, extra={"trigger": "epoch", "loss": avg_epoch_loss})

        # Save latest checkpoint at completion if directory specified
        if self.checkpoint_dir:
            latest_path = os.path.join(self.checkpoint_dir, "checkpoint_latest.json")
            self.save(latest_path, extra={"trigger": "latest", "loss": history[-1]["loss"] if history else 0.0})

        return {
            "epochs_completed": self.current_epoch,
            "global_step": self.global_step,
            "stopped_early": self._stop_requested,
            "history": history,
        }

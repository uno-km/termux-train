"""
termux_train.runtime
====================
Mobile Execution Runtime & Safe Checkpointing Subsystem.
Provides atomic crash-resilient checkpointing, rollback guarantees, and MobileTrainer lifecycle management.
"""

from .checkpoint import (
    save_checkpoint,
    load_checkpoint,
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointSchemaError,
    CheckpointRollbackError,
)
from .trainer import MobileTrainer

__all__ = [
    "save_checkpoint",
    "load_checkpoint",
    "CheckpointError",
    "CheckpointIntegrityError",
    "CheckpointSchemaError",
    "CheckpointRollbackError",
    "MobileTrainer",
]

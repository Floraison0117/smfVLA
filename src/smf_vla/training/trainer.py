"""
Trainer for smfVLA.

This module implements the training loop for fine-tuning the action head
with few-NFE objectives.
"""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Model
    action_dim: int = 7
    action_horizon: int = 10
    hidden_dim: int = 256
    num_layers: int = 4

    # Training
    batch_size: int = 64
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    max_epochs: int = 100
    warmup_steps: int = 1000
    gradient_clip_norm: float = 1.0

    # NFE settings
    target_nfe: int = 1  # Target number of NFE for distillation
    teacher_nfe: int = 10  # Teacher model NFE for distillation

    # Paths
    data_dir: str = "data/libero"
    checkpoint_dir: str = "checkpoints/finetuned"
    log_dir: str = "logs/train"

    # Misc
    seed: int = 42
    num_workers: int = 4
    save_every: int = 10  # Save checkpoint every N epochs
    log_every: int = 100  # Log every N steps

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


class SmfVLATrainer:
    """
    Trainer for smfVLA action head fine-tuning.

    Supports:
    1. Standard flow matching training
    2. Consistency distillation (teacher-student)
    3. Progressive distillation (multi-stage)
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: Optional[Any] = None,
        val_dataset: Optional[Any] = None,
        teacher_model: Optional[nn.Module] = None,
    ):
        self.model = model
        self.config = config
        self.teacher_model = teacher_model

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        if self.teacher_model is not None:
            self.teacher_model = self.teacher_model.to(self.device)
            self.teacher_model.eval()

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Setup scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.max_epochs,
            eta_min=config.learning_rate * 0.01,
        )

        # Setup data
        if train_dataset is not None:
            self.train_loader = DataLoader(
                train_dataset,
                batch_size=config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                pin_memory=True,
            )
        else:
            self.train_loader = None

        if val_dataset is not None:
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                pin_memory=True,
            )
        else:
            self.val_loader = None

        # Setup logging
        self.log_dir = Path(config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float("inf")

    def train(self):
        """Run the training loop."""
        logger.info(f"Starting training for {self.config.max_epochs} epochs")
        logger.info(f"Target NFE: {self.config.target_nfe}")
        logger.info(f"Device: {self.device}")

        for epoch in range(self.config.max_epochs):
            self.current_epoch = epoch
            train_loss = self.train_epoch()

            # Validation
            if self.val_loader is not None:
                val_loss = self.validate()
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best")
            else:
                logger.info(f"Epoch {epoch}: train_loss={train_loss:.4f}")

            # Save periodic checkpoint
            if (epoch + 1) % self.config.save_every == 0:
                self.save_checkpoint(f"epoch_{epoch+1}")

            self.scheduler.step()

        logger.info("Training completed")

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)

            total_loss += loss
            num_batches += 1
            self.global_step += 1

            if self.global_step % self.config.log_every == 0:
                logger.info(f"Step {self.global_step}: loss={loss:.4f}")

        return total_loss / max(num_batches, 1)

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Single training step."""
        # Move batch to device
        batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        # Forward pass
        loss = self.compute_loss(batch)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if self.config.gradient_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.gradient_clip_norm,
            )

        self.optimizer.step()

        return loss.item()

    def compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute training loss.

        Supports:
        1. Flow matching loss (standard training)
        2. Consistency distillation loss (teacher-student)
        """
        # Extract batch components
        actions = batch["actions"]  # [B, T, action_dim]
        context = batch["context"]  # [B, seq_len, hidden_dim]

        B = actions.shape[0]

        # Sample random timesteps
        t = torch.rand(B, 1, device=self.device)

        # Sample noise
        noise = torch.randn_like(actions)

        # Create noisy actions using flow matching interpolation
        noisy_actions = (1 - t) * noise + t * actions

        # Predict velocity field
        predicted_velocity = self.model(noisy_actions, t, context)

        # Target velocity (for flow matching: x1 - x0)
        target_velocity = actions - noise

        # Compute loss
        loss = nn.functional.mse_loss(predicted_velocity, target_velocity)

        # Add consistency distillation loss if teacher is available
        if self.teacher_model is not None and self.config.target_nfe == 1:
            with torch.no_grad():
                # Teacher prediction with more NFE
                teacher_actions = self.sample_teacher(context)

            # Student prediction with 1 NFE
            student_actions = self.model.sample_actions(context, num_steps=1)

            # Consistency loss
            consistency_loss = nn.functional.mse_loss(student_actions, teacher_actions)

            # Combine losses
            loss = loss + 0.1 * consistency_loss

        return loss

    @torch.no_grad()
    def sample_teacher(self, context: torch.Tensor) -> torch.Tensor:
        """Sample actions using teacher model with more NFE."""
        return self.teacher_model.sample_actions(
            context,
            num_steps=self.config.teacher_nfe,
        )

    def validate(self) -> float:
        """Run validation."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                loss = self.compute_loss(batch)
                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(num_batches, 1)

    def save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "config": self.config.to_dict(),
        }

        path = self.checkpoint_dir / f"{name}.pt"
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]

        logger.info(f"Loaded checkpoint from {path}")

"""
Unit tests for the training pipeline module.

Covers single training step execution, loss decreasing over epochs,
gradient updates, and checkpoint saving/loading.
Run with:  pytest tests/test_training.py -v
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import AIModel, build_model
from models.train_model import Trainer


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def synthetic_data():
    """Create synthetic training and validation data for binary classification."""
    torch.manual_seed(42)
    X_train = torch.randn(200, 10)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).float().unsqueeze(1)
    X_val = torch.randn(60, 10)
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).float().unsqueeze(1)
    return X_train, y_train, X_val, y_val


@pytest.fixture
def data_loaders(synthetic_data):
    """Create DataLoaders from synthetic data."""
    X_train, y_train, X_val, y_val = synthetic_data
    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)
    return train_loader, val_loader


@pytest.fixture
def trainer(data_loaders):
    """Create a Trainer instance with synthetic data."""
    train_loader, val_loader = data_loaders
    model, optimizer, criterion = build_model(
        input_dim=10, hidden_dims=[32, 16], output_dim=1, learning_rate=1e-3
    )
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        device="cpu",
        checkpoint_dir=tempfile.mkdtemp(),
        patience=10,
        binary=True,
    )
    return trainer


# ===========================================================================
# Tests – Single Training Step
# ===========================================================================


class TestSingleTrainingStep:
    """Tests for executing a single training epoch."""

    def test_train_epoch_returns_loss_and_accuracy(self, trainer):
        """train_epoch should return (loss, accuracy) tuple."""
        loss, accuracy = trainer.train_epoch(epoch=1)
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0
        assert 0.0 <= accuracy <= 1.0

    def test_train_epoch_loss_is_finite(self, trainer):
        """Training loss should be a finite number (no NaN or Inf)."""
        loss, _ = trainer.train_epoch(epoch=1)
        assert np.isfinite(loss)

    def test_train_epoch_updates_parameters(self, trainer):
        """After one training epoch, model parameters should have changed."""
        # Capture parameters before training
        params_before = [p.data.clone() for p in trainer.model.parameters()]

        trainer.train_epoch(epoch=1)

        # Check that at least some parameters changed
        changed = sum(
            1 for before, after in zip(params_before, trainer.model.parameters())
            if not torch.equal(before, after.data)
        )
        assert changed > 0, "No model parameters were updated during training"

    def test_validate_returns_loss_and_accuracy(self, trainer):
        """validate should return (loss, accuracy) tuple."""
        loss, accuracy = trainer.validate()
        assert isinstance(loss, float)
        assert isinstance(accuracy, float)
        assert loss > 0
        assert 0.0 <= accuracy <= 1.0


# ===========================================================================
# Tests – Loss Decreases
# ===========================================================================


class TestLossDecrease:
    """Tests that the training loss decreases over multiple epochs."""

    def test_loss_decreases_over_epochs(self, trainer):
        """After several training epochs, the loss should generally decrease."""
        # Run 5 epochs
        initial_loss, _ = trainer.train_epoch(epoch=1)
        final_loss = initial_loss

        for epoch in range(2, 6):
            final_loss, _ = trainer.train_epoch(epoch=epoch)

        # With synthetic data and a learnable model, loss should decrease
        assert final_loss < initial_loss, (
            f"Loss did not decrease: initial={initial_loss:.6f}, final={final_loss:.6f}"
        )

    def test_training_loop_loss_history(self, trainer):
        """The train() method should populate history with decreasing loss trend."""
        history = trainer.train(num_epochs=10)
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 10
        assert len(history["val_loss"]) == 10

        # The last few epochs should have lower loss than the first few (on average)
        early_avg = np.mean(history["train_loss"][:3])
        late_avg = np.mean(history["train_loss"][-3:])
        assert late_avg < early_avg, (
            f"Training loss did not trend downward: early_avg={early_avg:.6f}, late_avg={late_avg:.6f}"
        )


# ===========================================================================
# Tests – Checkpoint Saving / Loading
# ===========================================================================


class TestCheckpointSaving:
    """Tests for model checkpoint saving and loading."""

    def test_checkpoint_file_created(self, trainer):
        """save_checkpoint should create a .pt file in the checkpoint directory."""
        trainer.save_checkpoint(epoch=1, val_loss=0.5)
        ckpt_path = os.path.join(trainer.checkpoint_dir, "best_model.pt")
        assert os.path.isfile(ckpt_path)

    def test_checkpoint_file_contains_state_dict(self, trainer):
        """The saved checkpoint should contain model_state_dict."""
        trainer.save_checkpoint(epoch=3, val_loss=0.3)
        ckpt_path = os.path.join(trainer.checkpoint_dir, "best_model.pt")
        state = torch.load(ckpt_path, map_location="cpu")
        assert "model_state_dict" in state
        assert "optimizer_state_dict" in state
        assert "epoch" in state
        assert state["epoch"] == 3
        assert "val_loss" in state
        assert state["val_loss"] == 0.3

    def test_custom_checkpoint_filename(self, trainer):
        """save_checkpoint with a custom filename should use that name."""
        trainer.save_checkpoint(epoch=5, val_loss=0.1, filename="epoch5.pt")
        ckpt_path = os.path.join(trainer.checkpoint_dir, "epoch5.pt")
        assert os.path.isfile(ckpt_path)

    def test_load_checkpoint_restores_weights(self, trainer):
        """Loading a checkpoint should restore the model's weights exactly."""
        # Save a checkpoint
        trainer.train_epoch(epoch=1)
        trainer.save_checkpoint(epoch=1, val_loss=0.5)

        # Record current weights
        weights_before = {name: p.data.clone() for name, p in trainer.model.named_parameters()}

        # Reinitialize the model and load the checkpoint
        model2, _, _ = build_model(
            input_dim=10, hidden_dims=[32, 16], output_dim=1, learning_rate=1e-3, device="cpu"
        )
        ckpt = torch.load(os.path.join(trainer.checkpoint_dir, "best_model.pt"), map_location="cpu")
        model2.load_state_dict(ckpt["model_state_dict"])

        for name, param in model2.named_parameters():
            torch.testing.assert_close(param.data, weights_before[name])


# ===========================================================================
# Tests – Early Stopping
# ===========================================================================


class TestEarlyStopping:
    """Tests for early stopping behavior."""

    def test_early_stopping_triggers(self, trainer):
        """With low patience, training should stop early on synthetic data."""
        # Patience is set to 2 to trigger early stopping quickly
        trainer.patience = 2
        trainer.epochs_no_improve = 0
        trainer.best_val_loss = float("-inf")  # Start with extremely low loss

        # Manually set best_val_loss very low so improvement is unlikely
        trainer.best_val_loss = -1e10
        history = trainer.train(num_epochs=20)

        # Should have stopped before reaching 20 epochs
        assert len(history["train_loss"]) < 20, (
            f"Early stopping did not trigger; ran {len(history['train_loss'])} epochs"
        )

    def test_no_early_stopping_when_improving(self, trainer):
        """With high patience, training should complete all epochs on learnable data."""
        trainer.patience = 100
        history = trainer.train(num_epochs=5)
        assert len(history["train_loss"]) == 5

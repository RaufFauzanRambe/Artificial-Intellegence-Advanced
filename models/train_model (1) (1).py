"""
Model Training Module

Provides the Trainer class for managing the full training loop including
validation, learning rate scheduling, early stopping, and model checkpointing.
"""

import os
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class Trainer:
    """
    Handles training and validation of a PyTorch model with early stopping.

    The trainer manages:
    - Iteration over epochs and batches
    - Forward/backward passes with gradient clipping
    - Validation at the end of each epoch
    - Learning rate scheduling
    - Early stopping based on validation loss
    - Saving the best model checkpoint

    Args:
        model (nn.Module): The PyTorch model to train.
        train_loader (DataLoader): DataLoader for the training set.
        val_loader (DataLoader): DataLoader for the validation set.
        optimizer: PyTorch optimizer instance.
        loss_fn: Loss function (e.g., nn.BCEWithLogitsLoss, nn.CrossEntropyLoss).
        scheduler (optional): Learning rate scheduler. Defaults to None.
        device (str): Device to use for training. Defaults to 'auto' (GPU if available).
        checkpoint_dir (str): Directory to save model checkpoints. Defaults to 'checkpoints/'.
        patience (int): Number of epochs to wait for improvement before early stopping. Defaults to 10.
        binary (bool): Whether this is a binary classification task. Defaults to True.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer,
        loss_fn: nn.Module,
        scheduler=None,
        device: str = "auto",
        checkpoint_dir: str = "checkpoints",
        patience: int = 10,
        binary: bool = True,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.loss_fn = loss_fn.to(self.device)
        self.scheduler = scheduler
        self.checkpoint_dir = checkpoint_dir
        self.patience = patience
        self.binary = binary

        # Training state
        self.best_val_loss = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0
        self.history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

        # Create checkpoint directory if it does not exist
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def train_epoch(self, epoch: int) -> tuple:
        """
        Run one full epoch of training.

        Args:
            epoch (int): Current epoch number (used for logging).

        Returns:
            tuple: (average_loss, accuracy) for this epoch.
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(inputs)
            loss = self.loss_fn(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # Update weights
            self.optimizer.step()

            # Track metrics
            running_loss += loss.item() * inputs.size(0)
            total += targets.size(0)

            if self.binary:
                predictions = (torch.sigmoid(outputs) >= 0.5).float()
            else:
                predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == targets).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = correct / total

        return epoch_loss, epoch_acc

    def validate(self) -> tuple:
        """
        Evaluate the model on the validation set.

        Returns:
            tuple: (average_loss, accuracy) on the validation set.
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)

                running_loss += loss.item() * inputs.size(0)
                total += targets.size(0)

                if self.binary:
                    predictions = (torch.sigmoid(outputs) >= 0.5).float()
                else:
                    predictions = torch.argmax(outputs, dim=1)

                correct += (predictions == targets).sum().item()

        val_loss = running_loss / total
        val_acc = correct / total

        return val_loss, val_acc

    def save_checkpoint(self, epoch: int, val_loss: float, filename: Optional[str] = None):
        """
        Save a model checkpoint.

        Args:
            epoch (int): Current epoch number.
            val_loss (float): Validation loss at this epoch.
            filename (str, optional): Custom checkpoint filename. Defaults to 'best_model.pt'.
        """
        if filename is None:
            filename = "best_model.pt"

        checkpoint_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "history": self.history,
            },
            checkpoint_path,
        )

    def train(self, num_epochs: int) -> dict:
        """
        Run the complete training loop.

        Iterates over the specified number of epochs, performing training and
        validation at each step. Implements early stopping based on validation loss
        and saves the best model checkpoint.

        Args:
            num_epochs (int): Maximum number of epochs to train.

        Returns:
            dict: Training history containing 'train_loss', 'val_loss',
                  'train_acc', and 'val_acc' lists for each epoch.
        """
        print(f"Training on device: {self.device}")
        print(f"{'Epoch':>5} | {'Train Loss':>12} | {'Val Loss':>10} | "
              f"{'Train Acc':>10} | {'Val Acc':>8} | {'Time':>8}")
        print("-" * 72)

        start_time = time.time()

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            train_loss, train_acc = self.train_epoch(epoch)
            val_loss, val_acc = self.validate()

            # Update learning rate if scheduler is provided
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_acc"].append(val_acc)

            epoch_time = time.time() - epoch_start

            print(
                f"{epoch:>5d} | {train_loss:>12.6f} | {val_loss:>10.6f} | "
                f"{train_acc:>10.4f} | {val_acc:>8.4f} | {epoch_time:>7.2f}s"
            )

            # Check for improvement and save checkpoint
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, val_loss)
                print(f"  -> Best model saved (val_loss: {val_loss:.6f})")
            else:
                self.epochs_no_improve += 1
                print(f"  -> No improvement for {self.epochs_no_improve}/{self.patience} epochs")

            # Early stopping check
            if self.epochs_no_improve >= self.patience:
                print(f"\nEarly stopping triggered after {epoch} epochs. "
                      f"Best epoch was {self.best_epoch} with val_loss={self.best_val_loss:.6f}")
                break

        total_time = time.time() - start_time
        print(f"\nTraining complete in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.6f} at epoch {self.best_epoch}")

        return self.history


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from model import build_model
    from torch.utils.data import TensorDataset

    # Create synthetic data for demonstration
    torch.manual_seed(42)
    X_train = torch.randn(400, 5)
    y_train = (X_train[:, 0] + X_train[:, 1] > 0).float().unsqueeze(1)
    X_val = torch.randn(100, 5)
    y_val = (X_val[:, 0] + X_val[:, 1] > 0).float().unsqueeze(1)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    model, optimizer, criterion = build_model(
        input_dim=5, hidden_dims=[64, 32], output_dim=1, learning_rate=1e-3
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=criterion,
        scheduler=scheduler,
        checkpoint_dir="checkpoints",
        patience=10,
    )

    history = trainer.train(num_epochs=50)

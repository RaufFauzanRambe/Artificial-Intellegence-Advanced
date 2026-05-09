"""
Model Inference Module

Provides the ModelInference class for loading a trained model from a checkpoint
and running predictions on individual samples or batches of data.
"""

import os
from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn


class ModelInference:
    """
    Loads a trained model and provides methods for running inference.

    Handles model loading, input preprocessing, single-sample and batch
    predictions, and returns results with confidence scores.

    Args:
        model (nn.Module): The model class (not an instance) to instantiate.
        checkpoint_path (str): Path to the saved model checkpoint (.pt file).
        input_dim (int): Number of input features expected by the model.
        hidden_dims (list, optional): Hidden layer dimensions. Default [128, 64, 32].
        output_dim (int): Number of output classes. Default 1.
        device (str): Device for inference. Defaults to 'auto'.
    """

    def __init__(
        self,
        model_class: type,
        checkpoint_path: str,
        input_dim: int,
        hidden_dims: Optional[list] = None,
        output_dim: int = 1,
        device: str = "auto",
    ):
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_class = model_class
        self.checkpoint_path = checkpoint_path
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.binary = output_dim == 1

        # Initialize and load the model
        self.model = self._build_model()
        self.model = self.load_model(checkpoint_path)

    def _build_model(self) -> nn.Module:
        """
        Instantiate a fresh model with the configured architecture.

        Returns:
            nn.Module: Model instance moved to the target device.
        """
        model = self.model_class(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=self.output_dim,
        ).to(self.device)
        return model

    def load_model(self, checkpoint_path: str) -> nn.Module:
        """
        Load model weights from a checkpoint file.

        Args:
            checkpoint_path (str): Path to the .pt checkpoint file.

        Returns:
            nn.Module: Model with loaded weights, set to evaluation mode.

        Raises:
            FileNotFoundError: If the checkpoint file does not exist.
            RuntimeError: If the checkpoint cannot be loaded.
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        epoch = checkpoint.get("epoch", "unknown")
        val_loss = checkpoint.get("val_loss", "unknown")
        print(f"Model loaded from {checkpoint_path} (epoch={epoch}, val_loss={val_loss})")

        return self.model

    @staticmethod
    def preprocess_input(
        raw_input: Union[list, np.ndarray],
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
    ) -> torch.Tensor:
        """
        Preprocess raw input data into a normalized PyTorch tensor.

        Converts the input to float32, optionally standardizes using the
        provided mean and standard deviation, and wraps it as a tensor.

        Args:
            raw_input (list or np.ndarray): Raw feature values of shape
                (input_dim,) for a single sample or (batch_size, input_dim)
                for a batch.
            mean (np.ndarray, optional): Mean values for each feature (standardization).
            std (np.ndarray, optional): Standard deviation for each feature.

        Returns:
            torch.Tensor: Preprocessed tensor of shape (1, input_dim) or (batch_size, input_dim).
        """
        # Convert to numpy array if needed
        if isinstance(raw_input, list):
            arr = np.array(raw_input, dtype=np.float32)
        else:
            arr = np.array(raw_input, dtype=np.float32)

        # Ensure 2D shape: (batch_size, input_dim)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Standardize if mean and std are provided
        if mean is not None and std is not None:
            arr = (arr - mean) / (std + 1e-8)

        return torch.tensor(arr, dtype=torch.float32)

    def predict(self, sample: Union[list, np.ndarray], threshold: float = 0.5) -> dict:
        """
        Run prediction on a single sample.

        Args:
            sample (list or np.ndarray): Feature values for one sample.
            threshold (float): Classification threshold for binary tasks. Default 0.5.

        Returns:
            dict: Prediction result containing:
                - 'predicted_class': int, the predicted class label
                - 'confidence': float, the prediction confidence score
                - 'probabilities': np.ndarray, raw probability outputs
        """
        input_tensor = self.preprocess_input(sample).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

            if self.binary:
                prob = torch.sigmoid(output).item()
                predicted_class = 1 if prob >= threshold else 0
                confidence = prob if predicted_class == 1 else (1.0 - prob)
                probabilities = np.array([1.0 - prob, prob])
            else:
                probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()
                predicted_class = int(np.argmax(probs))
                confidence = float(probs[predicted_class])
                probabilities = probs

        return {
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
            "probabilities": np.round(probabilities, 4),
        }

    def predict_batch(
        self, batch: Union[list, np.ndarray], threshold: float = 0.5
    ) -> dict:
        """
        Run predictions on a batch of samples.

        Args:
            batch (list or np.ndarray): Feature values of shape (batch_size, input_dim).
            threshold (float): Classification threshold for binary tasks. Default 0.5.

        Returns:
            dict: Batch prediction results containing:
                - 'predicted_classes': list of int, predicted class labels
                - 'confidences': list of float, confidence scores
                - 'probabilities': np.ndarray, probability matrix (batch_size x num_classes)
        """
        input_tensor = self.preprocess_input(batch).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)

            if self.binary:
                probs = torch.sigmoid(output).squeeze().cpu().numpy()
                if probs.ndim == 0:
                    probs = np.array([probs.item()])
                predicted_classes = (probs >= threshold).astype(int).tolist()
                confidences = np.where(
                    np.array(predicted_classes) == 1, probs, 1.0 - probs
                ).tolist()
                prob_matrix = np.column_stack([1.0 - probs, probs])
            else:
                prob_matrix = torch.softmax(output, dim=1).cpu().numpy()
                predicted_classes = np.argmax(prob_matrix, axis=1).tolist()
                confidences = [float(prob_matrix[i, c]) for i, c in enumerate(predicted_classes)]

        return {
            "predicted_classes": predicted_classes,
            "confidences": [round(c, 4) for c in confidences],
            "probabilities": np.round(prob_matrix, 4),
        }


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from model import AIModel

    checkpoint = "checkpoints/best_model.pt"

    if not os.path.exists(checkpoint):
        print(f"Checkpoint not found at {checkpoint}.")
        print("Training a quick model for demonstration...")
        from train_model import Trainer
        from torch.utils.data import DataLoader, TensorDataset

        torch.manual_seed(42)
        X = torch.randn(300, 5)
        y = (X[:, 0] + X[:, 1] > 0).float().unsqueeze(1)
        dataset = TensorDataset(X, y)
        loader = DataLoader(dataset, batch_size=32)

        model = AIModel(input_dim=5, hidden_dims=[64, 32], output_dim=1)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            optimizer=optimizer,
            loss_fn=criterion,
            checkpoint_dir="checkpoints",
            patience=5,
        )
        trainer.train(num_epochs=20)

    # Set up inference
    inferencer = ModelInference(
        model_class=AIModel,
        checkpoint_path="checkpoints/best_model.pt",
        input_dim=5,
        hidden_dims=[64, 32],
        output_dim=1,
    )

    # Single prediction
    sample = [3.5, 7.8, 2.1, 0.9, 5.2]
    result = inferencer.predict(sample)
    print(f"\nSingle prediction:")
    print(f"  Input:    {sample}")
    print(f"  Class:    {result['predicted_class']}")
    print(f"  Conf:     {result['confidence']}")
    print(f"  Probs:    {result['probabilities']}")

    # Batch prediction
    batch = [
        [3.5, 7.8, 2.1, 0.9, 5.2],
        [0.8, 3.1, 0.4, 0.2, 1.5],
        [5.9, 9.1, 3.8, 1.6, 7.8],
    ]
    batch_result = inferencer.predict_batch(batch)
    print(f"\nBatch predictions:")
    for i, (inp, pred, conf) in enumerate(
        zip(batch, batch_result["predicted_classes"], batch_result["confidences"])
    ):
        print(f"  Sample {i}: class={pred}, confidence={conf}, input={inp}")

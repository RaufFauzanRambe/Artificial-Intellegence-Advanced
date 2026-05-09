"""
Model Evaluation Module

Provides functions and the ModelEvaluator class for computing classification
metrics, generating reports, and saving evaluation results to disk.
"""

import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard classification metrics given ground truth and predictions.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        dict: Dictionary containing accuracy, precision, recall, and F1 score.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels: list = None
) -> dict:
    """
    Compute the confusion matrix and return it as a dictionary.

    Args:
        y_true (np.ndarray): Ground truth labels.
        y_pred (np.ndarray): Predicted labels.
        labels (list, optional): List of label names. Defaults to None.

    Returns:
        dict: Dictionary with 'matrix' (2D list) and 'labels' keys.
    """
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "matrix": cm.tolist(),
        "labels": labels if labels else sorted(set(y_true.tolist())),
    }


class ModelEvaluator:
    """
    Evaluates a trained PyTorch model on a given dataset.

    Provides methods to run inference, compute metrics, generate a
    classification report, and persist results to JSON and text files.

    Args:
        model (nn.Module): Trained PyTorch model.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        device (str): Device to run evaluation on. Defaults to 'auto'.
        binary (bool): Whether this is a binary classification task. Defaults to True.
    """

    def __init__(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        device: str = "auto",
        binary: bool = True,
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        self.data_loader = data_loader
        self.binary = binary

    def evaluate(self) -> dict:
        """
        Run full evaluation on the dataset.

        Returns:
            dict: Dictionary containing:
                - 'predictions': numpy array of predicted labels
                - 'probabilities': numpy array of predicted probabilities
                - 'true_labels': numpy array of ground truth labels
                - 'metrics': dict of accuracy, precision, recall, F1
                - 'confusion_matrix': dict with matrix and labels
        """
        all_probs = []
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.data_loader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)

                outputs = self.model(inputs)

                if self.binary:
                    probs = torch.sigmoid(outputs).squeeze()
                    preds = (probs >= 0.5).float()
                else:
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probs, dim=1)

                all_probs.append(probs.cpu().numpy())
                all_preds.append(preds.cpu().numpy())
                all_targets.append(targets.cpu().numpy())

        y_true = np.concatenate(all_targets).flatten()
        y_pred = np.concatenate(all_preds).flatten()
        y_prob = np.concatenate(all_probs).flatten()

        metrics = compute_metrics(y_true, y_pred)
        cm = compute_confusion_matrix(y_true, y_pred)

        return {
            "predictions": y_pred,
            "probabilities": y_prob,
            "true_labels": y_true,
            "metrics": metrics,
            "confusion_matrix": cm,
        }

    def classification_report(
        self, target_names: Optional[list] = None, digits: int = 4
    ) -> str:
        """
        Generate a human-readable classification report.

        Args:
            target_names (list, optional): Names for each class. Defaults to None.
            digits (int): Number of decimal places for metrics. Defaults to 4.

        Returns:
            str: The classification report as a formatted string.
        """
        results = self.evaluate()
        report = classification_report(
            results["true_labels"],
            results["predictions"],
            target_names=target_names,
            digits=digits,
        )
        return report

    def save_results(self, output_dir: str = "results") -> None:
        """
        Save evaluation results to JSON and text files.

        Creates the following files in the output directory:
        - metrics.json: All computed metrics and confusion matrix
        - classification_report.txt: Human-readable classification report

        Args:
            output_dir (str): Directory to save results. Defaults to 'results'.
        """
        os.makedirs(output_dir, exist_ok=True)

        # Run evaluation
        results = self.evaluate()

        # Save metrics to JSON
        metrics_data = {
            "metrics": results["metrics"],
            "confusion_matrix": results["confusion_matrix"],
        }

        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics_data, f, indent=4)
        print(f"Metrics saved to {metrics_path}")

        # Save classification report to text file
        report_path = os.path.join(output_dir, "classification_report.txt")
        report = classification_report(
            results["true_labels"],
            results["predictions"],
            target_names=["Class 0", "Class 1"],
            digits=4,
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(f"Classification report saved to {report_path}")

        # Print summary
        print("\n--- Evaluation Summary ---")
        for metric_name, value in results["metrics"].items():
            print(f"  {metric_name:>12s}: {value:.4f}")
        print(f"\nConfusion Matrix:")
        cm = np.array(results["confusion_matrix"]["matrix"])
        print(f"  {cm[0]}")
        print(f"  {cm[1]}")


# ---------------------------------------------------------------------------
# Example usage (run as script)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import sys

    from model import AIModel
    from torch.utils.data import TensorDataset

    parser = argparse.ArgumentParser(description="Evaluate a trained AI model")
    parser.add_argument(
        "--checkpoint", type=str, default="checkpoints/best_model.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save evaluation results"
    )
    args = parser.parse_args()

    # Create synthetic test data for demonstration
    torch.manual_seed(99)
    X_test = torch.randn(100, 5)
    y_test = (X_test[:, 0] + X_test[:, 1] > 0).float().unsqueeze(1)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Load model
    model = AIModel(input_dim=5, hidden_dims=[64, 32], output_dim=1)

    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        print(f"Warning: No checkpoint found at {args.checkpoint}, evaluating untrained model")

    evaluator = ModelEvaluator(model=model, data_loader=test_loader)
    evaluator.save_results(output_dir=args.output_dir)

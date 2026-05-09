"""
AI Model Definition Module

Provides the AIModel neural network class and a factory function for building models
with configurable architectures suitable for tabular classification tasks.
"""

import torch
import torch.nn as nn


class AIModel(nn.Module):
    """
    A fully-connected feedforward neural network for classification tasks.

    The architecture consists of:
    - An input layer matching the number of input features
    - N hidden layers with configurable dimensions, BatchNorm, ReLU activation, and dropout
    - An output layer matching the number of output classes

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list[int]): List of hidden layer sizes. Default is [128, 64, 32].
        output_dim (int): Number of output classes. Default is 1 (binary classification).
        dropout_rate (float): Dropout probability applied after each hidden layer. Default is 0.3.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list = None,
        output_dim: int = 1,
        dropout_rate: float = 0.3,
    ):
        super(AIModel, self).__init__()

        if hidden_dims is None:
            hidden_dims = [128, 64, 32]

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Build the layers dynamically based on hidden_dims
        layers = []
        prev_dim = input_dim

        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_rate))
            prev_dim = hidden_dim

        # Output layer (no activation for binary classification with BCEWithLogitsLoss,
        # or use nn.Sigmoid() if using BCELoss)
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights using Kaiming (He) initialization for ReLU
        self._initialize_weights()

    def _initialize_weights(self):
        """Apply Kaiming normal initialization to linear layers and default init to batch norms."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, output_dim).
        """
        return self.network(x)

    def get_num_parameters(self) -> dict:
        """
        Return the total and trainable parameter counts.

        Returns:
            dict: Dictionary with 'total' and 'trainable' parameter counts.
        """
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable}


def build_model(
    input_dim: int,
    hidden_dims: list = None,
    output_dim: int = 1,
    dropout_rate: float = 0.3,
    learning_rate: float = 1e-3,
    device: str = None,
) -> tuple:
    """
    Factory function to build and return a model, optimizer, and loss function.

    This is a convenience function that creates a model, moves it to the
    appropriate device, sets up an Adam optimizer, and selects a suitable
    loss function based on the output dimension.

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list[int], optional): Hidden layer sizes. Default [128, 64, 32].
        output_dim (int): Number of output classes. Default 1.
        dropout_rate (float): Dropout probability. Default 0.3.
        learning_rate (float): Learning rate for the optimizer. Default 1e-3.
        device (str, optional): Device to place the model on ('cpu' or 'cuda').
            If None, automatically selects CUDA if available.

    Returns:
        tuple: (model, optimizer, criterion) where:
            - model: The AIModel instance on the specified device.
            - optimizer: An Adam optimizer.
            - criterion: Loss function (BCEWithLogitsLoss for binary, CrossEntropyLoss for multi-class).
    """
    if hidden_dims is None:
        hidden_dims = [128, 64, 32]

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AIModel(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        output_dim=output_dim,
        dropout_rate=dropout_rate,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

    if output_dim == 1:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    return model, optimizer, criterion

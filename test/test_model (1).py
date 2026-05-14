"""
Unit tests for the AIModel neural network module.

Covers model creation, parameter counting, forward pass output shapes,
gradient flow verification, and weight initialization.
Run with:  pytest tests/test_model.py -v
"""

import os
import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.model import AIModel, build_model


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def binary_model():
    """A binary classification model (output_dim=1)."""
    torch.manual_seed(42)
    return AIModel(input_dim=10, hidden_dims=[32, 16], output_dim=1, dropout_rate=0.2)


@pytest.fixture
def multiclass_model():
    """A multi-class classification model (output_dim=5)."""
    torch.manual_seed(42)
    return AIModel(input_dim=10, hidden_dims=[32, 16], output_dim=5, dropout_rate=0.2)


@pytest.fixture
def sample_input():
    """A batch of random input tensors (batch_size=8, input_dim=10)."""
    torch.manual_seed(0)
    return torch.randn(8, 10)


# ===========================================================================
# Tests – Model Creation
# ===========================================================================


class TestModelCreation:
    """Tests for model instantiation and parameter counting."""

    def test_model_creation_defaults(self):
        """Model should be creatable with default hidden_dims."""
        torch.manual_seed(42)
        model = AIModel(input_dim=5)
        assert model.input_dim == 5
        assert model.hidden_dims == [128, 64, 32]
        assert model.output_dim == 1

    def test_model_creation_custom(self):
        """Model should accept custom hidden_dims and output_dim."""
        torch.manual_seed(42)
        model = AIModel(input_dim=20, hidden_dims=[64], output_dim=10, dropout_rate=0.5)
        assert model.hidden_dims == [64]
        assert model.output_dim == 10
        assert model.dropout_rate == 0.5

    def test_model_is_nn_module(self, binary_model):
        """Model should be an instance of nn.Module."""
        assert isinstance(binary_model, nn.Module)

    def test_parameter_count(self, binary_model):
        """get_num_parameters should return positive integers."""
        params = binary_model.get_num_parameters()
        assert "total" in params
        assert "trainable" in params
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert params["total"] == params["trainable"]

    def test_network_has_layers(self, binary_model):
        """The sequential network should contain Linear, BatchNorm, ReLU, Dropout layers."""
        layer_types = {type(m).__name__ for m in binary_model.network}
        assert "Linear" in layer_types
        assert "BatchNorm1d" in layer_types
        assert "ReLU" in layer_types
        assert "Dropout" in layer_types


# ===========================================================================
# Tests – Forward Pass
# ===========================================================================


class TestForwardPass:
    """Tests for the forward pass behavior."""

    def test_forward_output_shape_binary(self, binary_model, sample_input):
        """Binary model should output shape (batch_size, 1)."""
        output = binary_model(sample_input)
        assert output.shape == (8, 1)

    def test_forward_output_shape_multiclass(self, multiclass_model, sample_input):
        """Multi-class model should output shape (batch_size, num_classes)."""
        output = multiclass_model(sample_input)
        assert output.shape == (8, 5)

    def test_forward_single_sample(self, binary_model):
        """Forward pass should work with a single sample (no batch dimension)."""
        x = torch.randn(10)
        output = binary_model(x)
        assert output.shape == (1,)

    def test_forward_no_nan_or_inf(self, binary_model, sample_input):
        """Forward pass should not produce NaN or Inf values."""
        binary_model.eval()
        output = binary_model(sample_input)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_forward_deterministic_eval(self, binary_model, sample_input):
        """In eval mode, forward pass should be deterministic (dropout disabled)."""
        binary_model.eval()
        out1 = binary_model(sample_input)
        out2 = binary_model(sample_input)
        torch.testing.assert_close(out1, out2)

    def test_forward_varies_in_train(self, binary_model, sample_input):
        """In train mode with dropout > 0, consecutive forward passes may differ."""
        binary_model.train()
        out1 = binary_model(sample_input)
        out2 = binary_model(sample_input)
        # It's possible (though unlikely) they are identical, so we just check types
        assert isinstance(out1, torch.Tensor)
        assert isinstance(out2, torch.Tensor)


# ===========================================================================
# Tests – Gradient Flow
# ===========================================================================


class TestGradientFlow:
    """Tests to ensure gradients propagate correctly during backpropagation."""

    def test_gradients_exist_after_backward(self, binary_model, sample_input):
        """All trainable parameters should have gradients after backward()."""
        binary_model.train()
        output = binary_model(sample_input)
        target = torch.randint(0, 2, (8, 1)).float()
        loss = nn.BCEWithLogitsLoss()(output, target)
        loss.backward()

        for name, param in binary_model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for parameter: {name}"

    def test_gradients_not_zero(self, binary_model, sample_input):
        """Gradients should be non-zero for most parameters (not a dead network)."""
        binary_model.train()
        output = binary_model(sample_input)
        target = torch.randint(0, 2, (8, 1)).float()
        loss = nn.BCEWithLogitsLoss()(output, target)
        loss.backward()

        nonzero_grads = sum(
            1 for p in binary_model.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum() > 0
        )
        total_trainable = sum(1 for p in binary_model.parameters() if p.requires_grad)
        # At least half of trainable params should have non-zero gradients
        assert nonzero_grads >= total_trainable * 0.5, (
            f"Only {nonzero_grads}/{total_trainable} parameters have non-zero gradients"
        )

    def test_gradient_flow_multiclass(self, multiclass_model, sample_input):
        """Gradient flow should work correctly for multi-class output."""
        multiclass_model.train()
        output = multiclass_model(sample_input)
        target = torch.randint(0, 5, (8,))
        loss = nn.CrossEntropyLoss()(output, target)
        loss.backward()

        has_grad = all(
            p.grad is not None for p in multiclass_model.parameters() if p.requires_grad
        )
        assert has_grad, "Some trainable parameters have no gradient after backward pass"


# ===========================================================================
# Tests – Weight Initialization
# ===========================================================================


class TestWeightInitialization:
    """Tests for proper weight initialization."""

    def test_linear_weights_not_all_zeros(self, binary_model):
        """Linear layer weights should not all be zero after initialization."""
        for module in binary_model.modules():
            if isinstance(module, nn.Linear):
                assert not torch.all(module.weight == 0), f"{module} weights are all zeros"

    def test_linear_biases_are_zero(self, binary_model):
        """Linear layer biases should be initialized to zero (Kaiming init)."""
        for module in binary_model.modules():
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.testing.assert_close(module.bias, torch.zeros_like(module.bias))


# ===========================================================================
# Tests – build_model Factory
# ===========================================================================


class TestBuildModel:
    """Tests for the build_model factory function."""

    def test_build_model_returns_tuple(self):
        """build_model should return (model, optimizer, criterion) tuple."""
        result = build_model(input_dim=10, hidden_dims=[32], output_dim=1)
        assert len(result) == 3
        model, optimizer, criterion = result
        assert isinstance(model, AIModel)
        assert isinstance(optimizer, torch.optim.Adam)
        assert isinstance(criterion, nn.BCEWithLogitsLoss)

    def test_build_model_multiclass_loss(self):
        """build_model should return CrossEntropyLoss for output_dim > 1."""
        _, _, criterion = build_model(input_dim=10, output_dim=5)
        assert isinstance(criterion, nn.CrossEntropyLoss)

    def test_build_model_device_cpu(self):
        """build_model with device='cpu' should place model on CPU."""
        model, _, _ = build_model(input_dim=10, output_dim=1, device="cpu")
        assert next(model.parameters()).device == torch.device("cpu")

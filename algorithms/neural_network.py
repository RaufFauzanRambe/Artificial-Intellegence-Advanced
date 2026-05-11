"""
Neural Network Module - Multi-Layer Perceptron Classifier from Scratch.

Implements a fully-connected feedforward neural network using only NumPy.
Supports arbitrary depth, He weight initialization, sigmoid activation,
backpropagation, and binary/categorical classification.

Example:
    >>> from neural_network import MLPClassifier
    >>> clf = MLPClassifier(layers=[2, 16, 8, 1], learning_rate=0.01, epochs=500)
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)
"""

import numpy as np


class MLPClassifier:
    """Multi-Layer Perceptron for binary classification (built from scratch).

    Attributes:
        layers (list[int]): Number of neurons in each layer including input.
        learning_rate (float): Step size for gradient descent.
        epochs (int): Number of full passes over the training data.
        weights (list[np.ndarray]): Weight matrices for each layer transition.
        biases (list[np.ndarray]): Bias vectors for each hidden and output layer.
    """

    def __init__(self, layers: list, learning_rate: float = 0.01, epochs: int = 1000):
        """Initialize the MLP architecture and parameters.

        Args:
            layers: List of integers specifying neurons per layer, e.g. [2, 16, 8, 1].
            learning_rate: Learning rate for gradient descent updates.
            epochs: Number of training iterations over the full dataset.
        """
        if len(layers) < 2:
            raise ValueError("At least 2 layers (input and output) are required.")
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using He initialization and biases to zero.

        He initialization sets weights ~ N(0, sqrt(2 / fan_in)), which is
        well-suited for networks using ReLU-like activations.  Biases are
        initialized to small positive values (0.01) to avoid dead neurons.
        """
        np.random.seed(42)
        for i in range(len(self.layers) - 1):
            fan_in = self.layers[i]
            fan_out = self.layers[i + 1]
            # He initialization: std = sqrt(2 / fan_in)
            std = np.sqrt(2.0 / fan_in)
            W = np.random.randn(fan_in, fan_out) * std
            b = np.zeros((1, fan_out)) + 0.01
            self.weights.append(W)
            self.biases.append(b)

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid activation function element-wise.

        Args:
            z: Pre-activation values (weighted sum + bias).

        Returns:
            Activations in the range (0, 1).
        """
        # Clip to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """Compute the derivative of the sigmoid function.

        Since sigmoid'(z) = sigmoid(z) * (1 - sigmoid(z)), we compute this
        directly from the activation output `a`.

        Args:
            a: Sigmoid activation output.

        Returns:
            Derivative of sigmoid at the given activations.
        """
        return a * (1.0 - a)

    def _forward_pass(self, X: np.ndarray) -> tuple:
        """Propagate input through the network layer by layer.

        Args:
            X: Input data of shape (n_samples, n_features).

        Returns:
            A tuple (activations, pre_activations) where:
                - activations: List of activation arrays for every layer.
                - pre_activations: List of z values (before activation) for
                  hidden and output layers.
        """
        activations = [X]
        pre_activations = []

        current = X
        for i in range(len(self.weights)):
            z = current @ self.weights[i] + self.biases[i]
            pre_activations.append(z)

            # Use sigmoid for hidden layers; sigmoid for output (binary classification)
            a = self._sigmoid(z)
            activations.append(a)
            current = a

        return activations, pre_activations

    def _backward_pass(self, activations: list, pre_activations: list, y: np.ndarray):
        """Compute gradients via backpropagation and update weights/biases.

        Uses binary cross-entropy loss derivative combined with sigmoid output.

        Args:
            activations: List of activation arrays from forward pass.
            pre_activations: List of z arrays from forward pass.
            y: Ground-truth labels of shape (n_samples, 1) or (n_samples,).
        """
        n = y.shape[0]
        y = y.reshape(-1, 1)

        # Output layer error: dL/dz = a_L - y (simplified for sigmoid + BCE loss)
        output_activation = activations[-1]
        delta = output_activation - y  # shape: (n_samples, 1)

        # Store per-layer gradients for potential batch accumulation
        grad_w = [None] * len(self.weights)
        grad_b = [None] * len(self.biases)

        # Traverse layers in reverse
        for i in range(len(self.weights) - 1, -1, -1):
            grad_w[i] = (activations[i].T @ delta) / n
            grad_b[i] = np.sum(delta, axis=0, keepdims=True) / n

            if i > 0:
                # Propagate error to previous layer
                delta = (delta @ self.weights[i].T) * self._sigmoid_derivative(activations[i])

            # Update weights and biases with gradient descent
            self.weights[i] -= self.learning_rate * grad_w[i]
            self.biases[i] -= self.learning_rate * grad_b[i]

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> "MLPClassifier":
        """Train the MLP on the given data using backpropagation.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,) or (n_samples, 1).
            verbose: If True, print loss every 100 epochs.

        Returns:
            self: The trained classifier.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).reshape(-1, 1)

        for epoch in range(self.epochs):
            activations, pre_activations = self._forward_pass(X)
            self._backward_pass(activations, pre_activations, y)

            if verbose and (epoch + 1) % 100 == 0:
                # Binary cross-entropy loss
                a = activations[-1]
                a = np.clip(a, 1e-8, 1 - 1e-8)
                loss = -np.mean(y * np.log(a) + (1 - y) * np.log(1 - a))
                print(f"Epoch {epoch + 1}/{self.epochs} - Loss: {loss:.6f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the given input.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Predicted labels as a 1-D numpy array of 0s and 1s.
        """
        X = np.asarray(X, dtype=np.float64)
        activations, _ = self._forward_pass(X)
        probs = activations[-1]
        return (probs >= 0.5).astype(int).flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities for the given input.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Probability estimates of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        activations, _ = self._forward_pass(X)
        return activations[-1].flatten()


if __name__ == "__main__":
    # --- Demo: XOR problem ---
    print("=" * 60)
    print("Neural Network - MLPClassifier Demo (XOR Problem)")
    print("=" * 60)

    # XOR dataset
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)
    y_xor = np.array([0, 1, 1, 0], dtype=np.float64)

    clf = MLPClassifier(layers=[2, 16, 8, 1], learning_rate=0.5, epochs=2000)
    clf.fit(X_xor, y_xor, verbose=True)

    predictions = clf.predict(X_xor)
    probabilities = clf.predict_proba(X_xor)

    print("\nXOR Predictions:")
    for i in range(len(X_xor)):
        print(f"  Input: {X_xor[i]} -> Pred: {predictions[i]} (prob={probabilities[i]:.4f})")
    accuracy = np.mean(predictions == y_xor)
    print(f"Accuracy: {accuracy:.2f}")

    # --- Demo: Non-linearly separable 2D data ---
    print("\n" + "=" * 60)
    print("Neural Network - MLPClassifier Demo (Circle Classification)")
    print("=" * 60)

    np.random.seed(0)
    X_circle = np.random.randn(200, 2)
    y_circle = (X_circle[:, 0] ** 2 + X_circle[:, 1] ** 2 < 1.0).astype(float)

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X_circle, y_circle, test_size=0.2, random_state=42
    )

    clf2 = MLPClassifier(layers=[2, 32, 16, 1], learning_rate=0.1, epochs=1000)
    clf2.fit(X_train, y_train, verbose=True)

    y_pred = clf2.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy on Circle Dataset: {acc:.4f}")

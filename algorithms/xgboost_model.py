"""
Gradient Boosting Module - Simplified XGBoost-style Classifier from Scratch.

Implements gradient boosting for binary classification using decision stumps
as weak learners. Uses log loss (binary cross-entropy) as the objective and
first/second-order gradient approximations for efficient tree construction.

This is a simplified version inspired by XGBoost, built entirely with NumPy.

Example:
    >>> from xgboost_model import GradientBoostingClassifier
    >>> clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)
"""

import numpy as np


class GradientBoostingClassifier:
    """Gradient Boosting Classifier for binary classification.

    Builds an additive ensemble of regression trees, where each new tree is
    fitted to the negative gradient (pseudo-residuals) of the loss function
    with respect to the current ensemble predictions.

    Uses log loss: L(y, F) = -[y * log(p) + (1-y) * log(1-p)]
    where p = sigmoid(F) and F is the raw log-odds prediction.

    Attributes:
        n_estimators: Number of boosting rounds (trees to add).
        learning_rate: Shrinkage factor applied to each new tree.
        max_depth: Maximum depth of each regression tree.
        min_samples_split: Minimum samples required to split a node.
        trees: List of fitted tree structures.
        initial_prediction: Starting log-odds prediction (log(prior)).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_split: int = 10,
        subsample: float = 1.0,
        random_state: int = 42,
    ):
        """Initialize the Gradient Boosting Classifier.

        Args:
            n_estimators: Number of boosting iterations.
            learning_rate: Shrinkage parameter (typically 0.01 to 0.3).
            max_depth: Maximum depth of each weak learner tree.
            min_samples_split: Minimum samples to split a node.
            subsample: Fraction of samples to use for each tree (stochastic GB).
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.initial_prediction = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Compute the sigmoid function, clipped for numerical stability.

        Args:
            z: Raw log-odds values.

        Returns:
            Probabilities in (0, 1).
        """
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _compute_gradients(self, y: np.ndarray, raw_predictions: np.ndarray) -> tuple:
        """Compute pseudo-residuals (negative gradient) and hessians for log loss.

        For binary cross-entropy loss with sigmoid:
            gradient_i = p_i - y_i
            hessian_i  = p_i * (1 - p_i)

        Args:
            y: True labels {0, 1} of shape (n_samples,).
            raw_predictions: Current raw predictions F(x) of shape (n_samples,).

        Returns:
            Tuple of (gradients, hessians), each of shape (n_samples,).
        """
        probs = self._sigmoid(raw_predictions)
        gradients = probs - y
        hessians = probs * (1.0 - probs)
        # Add small constant to hessians for numerical stability
        hessians = np.maximum(hessians, 1e-8)
        return gradients, hessians

    def _fit_tree_stump(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
    ) -> dict:
        """Fit a single regression tree using gradient and hessian information.

        Uses a simplified approach: builds a proper regression tree where leaf
        values are computed using Newton's method:
            leaf_value = -sum(gradients) / sum(hessians)

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            gradients: Pseudo-residuals of shape (n_samples,).
            hessians: Second-order gradients of shape (n_samples,).

        Returns:
            A dictionary representing the tree structure.
        """
        tree = self._build_tree(X, gradients, hessians, depth=0)
        return tree

    def _build_tree(
        self,
        X: np.ndarray,
        gradients: np.ndarray,
        hessians: np.ndarray,
        depth: int,
    ) -> dict:
        """Recursively build a regression tree for gradient boosting.

        Args:
            X: Feature matrix for current node samples.
            gradients: Gradients for current samples.
            hessians: Hessians for current samples.
            depth: Current tree depth.

        Returns:
            A dict with tree structure: {'leaf_value': val} or
            {'feature': idx, 'threshold': val, 'left': {...}, 'right': {...}}
        """
        n_samples = len(gradients)

        # Stopping criteria
        if (n_samples < self.min_samples_split
                or depth >= self.max_depth
                or n_samples <= 1):
            leaf_value = -np.sum(gradients) / np.sum(hessians)
            return {"leaf_value": leaf_value}

        # Find the best split using gain formula inspired by XGBoost:
        # Gain = 0.5 * [G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda)
        #               - (G_L + G_R)^2 / (H_L + H_R + lambda)] - gamma
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        best_left_mask = None

        G_total = np.sum(gradients)
        H_total = np.sum(hessians)
        reg_lambda = 1.0  # L2 regularization
        reg_gamma = 0.0   # Minimum gain to split

        n_features = X.shape[1]
        rng = np.random.default_rng(self.random_state)

        for feature_idx in range(n_features):
            values = X[:, feature_idx]
            thresholds = np.unique(values)

            # Limit number of thresholds for efficiency
            if len(thresholds) > 20:
                percentiles = np.linspace(0, 100, 21)
                thresholds = np.percentile(values, percentiles)
                thresholds = np.unique(thresholds)

            for threshold in thresholds:
                left_mask = values <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) < 1 or np.sum(right_mask) < 1:
                    continue

                G_L = np.sum(gradients[left_mask])
                H_L = np.sum(hessians[left_mask])
                G_R = np.sum(gradients[right_mask])
                H_R = np.sum(hessians[right_mask])

                gain = 0.5 * (
                    (G_L ** 2) / (H_L + reg_lambda)
                    + (G_R ** 2) / (H_R + reg_lambda)
                    - (G_total ** 2) / (H_total + reg_lambda)
                ) - reg_gamma

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_mask = left_mask

        # If no useful split found, create a leaf
        if best_feature is None:
            leaf_value = -G_total / H_total
            return {"leaf_value": leaf_value}

        left_mask = best_left_mask
        right_mask = ~best_left_mask

        return {
            "feature": best_feature,
            "threshold": best_threshold,
            "left": self._build_tree(X[left_mask], gradients[left_mask],
                                     hessians[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], gradients[right_mask],
                                      hessians[right_mask], depth + 1),
        }

    def _predict_tree(self, tree: dict, X: np.ndarray) -> np.ndarray:
        """Traverse a tree to get predictions for all samples.

        Args:
            tree: Tree dictionary structure.
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predictions of shape (n_samples,).
        """
        predictions = np.zeros(X.shape[0])

        for i in range(X.shape[0]):
            node = tree
            while "leaf_value" not in node:
                if X[i, node["feature"]] <= node["threshold"]:
                    node = node["left"]
                else:
                    node = node["right"]
            predictions[i] = node["leaf_value"]

        return predictions

    def fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = True) -> "GradientBoostingClassifier":
        """Train the gradient boosting classifier.

        Iteratively adds trees fitted to pseudo-residuals (gradients of the
        log loss) with respect to current predictions.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Binary labels of shape (n_samples,) in {0, 1}.
            verbose: If True, print training progress.

        Returns:
            self: The trained classifier.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)

        # Initial prediction: log-odds of the positive class prior
        pos_ratio = np.clip(np.mean(y), 1e-8, 1 - 1e-8)
        self.initial_prediction = np.log(pos_ratio / (1.0 - pos_ratio))

        # Initialize raw predictions with the base log-odds
        raw_predictions = np.full(n_samples, self.initial_prediction)
        self.trees = []

        for iteration in range(self.n_estimators):
            # Compute gradients and hessians
            gradients, hessians = self._compute_gradients(y, raw_predictions)

            # Optional: stochastic gradient boosting (subsample)
            if self.subsample < 1.0:
                sample_size = int(n_samples * self.subsample)
                idx = rng.choice(n_samples, size=sample_size, replace=False)
                X_sub = X[idx]
                g_sub = gradients[idx]
                h_sub = hessians[idx]
            else:
                X_sub = X
                g_sub = gradients
                h_sub = hessians

            # Fit a tree to the gradients/hessians
            tree = self._fit_tree_stump(X_sub, g_sub, h_sub)
            self.trees.append(tree)

            # Update raw predictions
            tree_predictions = self._predict_tree(tree, X)
            raw_predictions += self.learning_rate * tree_predictions

            if verbose and (iteration + 1) % 10 == 0:
                probs = self._sigmoid(raw_predictions)
                probs = np.clip(probs, 1e-8, 1 - 1e-8)
                loss = -np.mean(y * np.log(probs) + (1 - y) * np.log(1 - probs))
                print(f"  Iteration {iteration + 1:3d}/{self.n_estimators} - Log Loss: {loss:.6f}")

        return self

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Return raw log-odds predictions (sum of all tree outputs).

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Raw predictions of shape (n_samples,).
        """
        X = np.asarray(X, dtype=np.float64)
        raw = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            raw += self.learning_rate * self._predict_tree(tree, X)
        return raw

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Probability of class 1 for each sample, shape (n_samples,).
        """
        raw = self.predict_raw(X)
        return self._sigmoid(raw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels (0 or 1).

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Predicted labels as a 1-D numpy array.
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

    print("=" * 60)
    print("Gradient Boosting Classifier - Demo")
    print("=" * 60)

    # Generate a synthetic binary classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=3,
        n_classes=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test data:     {X_test.shape[0]} samples")
    print(f"Positive rate: {np.mean(y_train):.2%}")
    print()

    # Train gradient boosting classifier
    gb = GradientBoostingClassifier(
        n_estimators=50,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=10,
        subsample=0.8,
        random_state=42,
    )
    print("Training Gradient Boosting Classifier...")
    gb.fit(X_train, y_train, verbose=True)
    print("Training complete.\n")

    # Evaluate
    y_pred = gb.predict(X_test)
    y_proba = gb.predict_proba(X_test)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test AUC-ROC:  {auc:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Compare different hyperparameter settings
    print("\n" + "=" * 60)
    print("Hyperparameter Comparison")
    print("=" * 60)

    configs = [
        {"n_estimators": 20, "learning_rate": 0.3, "max_depth": 2},
        {"n_estimators": 50, "learning_rate": 0.1, "max_depth": 3},
        {"n_estimators": 100, "learning_rate": 0.05, "max_depth": 4},
    ]

    for cfg in configs:
        model = GradientBoostingClassifier(
            **cfg, min_samples_split=10, subsample=0.8, random_state=42
        )
        model.fit(X_train, y_train, verbose=False)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)
        a = accuracy_score(y_test, preds)
        auc_val = roc_auc_score(y_test, probs)
        print(f"  n_est={cfg['n_estimators']:3d}, lr={cfg['learning_rate']}, "
              f"depth={cfg['max_depth']} -> Acc={a:.4f}, AUC={auc_val:.4f}")

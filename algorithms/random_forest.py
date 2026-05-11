"""
Random Forest Module - Ensemble of Decision Trees from Scratch.

Implements a Random Forest classifier that builds multiple decorrelated decision
trees on bootstrap samples with random feature subsets. Uses majority voting
for final predictions. Depends on DecisionTree from decision_tree.py.

Example:
    >>> from random_forest import RandomForest
    >>> clf = RandomForest(n_estimators=100, max_depth=10, max_features='sqrt')
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)
"""

import numpy as np
from decision_tree import DecisionTree


class RandomForest:
    """Random Forest Classifier using bagging and feature subsampling.

    Each tree is trained on a bootstrap sample (sampling with replacement)
    and only considers a random subset of features at each split. This
    decorrelation between trees reduces variance and improves generalization.

    Attributes:
        n_estimators: Number of decision trees in the ensemble.
        max_depth: Maximum depth of each decision tree.
        max_features: Number or fraction of features to consider per split.
            - int: exact number of features
            - 'sqrt': square root of total features
            - 'log2': base-2 logarithm of total features
            - None: use all features
        min_samples_split: Minimum samples required to split a node.
        trees: List of trained DecisionTree instances.
        n_features: Total number of features seen during fit.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        max_features=None,
        min_samples_split: int = 2,
        random_state: int = 42,
    ):
        """Initialize the Random Forest.

        Args:
            n_estimators: Number of trees in the forest.
            max_depth: Maximum depth of each tree.
            max_features: Number/fraction of features to consider per split.
            min_samples_split: Minimum samples to split a node.
            random_state: Random seed for reproducibility.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.trees = []
        self.n_features = None

    def _resolve_max_features(self, n_features: int) -> int:
        """Resolve max_features to a concrete integer.

        Args:
            n_features: Total number of available features.

        Returns:
            Number of features to use at each split.
        """
        if self.max_features is None:
            return n_features
        if isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        if isinstance(self.max_features, str):
            if self.max_features == "sqrt":
                return max(1, int(np.sqrt(n_features)))
            elif self.max_features == "log2":
                return max(1, int(np.log2(n_features)))
        raise ValueError(
            f"Invalid max_features: {self.max_features}. "
            "Use an int, 'sqrt', 'log2', or None."
        )

    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Generate a bootstrap sample by sampling with replacement.

        Also returns out-of-bag (OOB) indices for potential OOB evaluation.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,).

        Returns:
            A tuple (X_bootstrap, y_bootstrap, oob_indices) where oob_indices
            are the sample indices NOT included in the bootstrap.
        """
        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_state)
        indices = rng.choice(n_samples, size=n_samples, replace=True)
        oob_mask = np.ones(n_samples, dtype=bool)
        oob_mask[indices] = False
        oob_indices = np.where(oob_mask)[0]

        return X[indices], y[indices], oob_indices

    def _get_random_features(self, n_total_features: int, rng: np.random.Generator) -> np.ndarray:
        """Randomly select feature indices for a split candidate.

        Args:
            n_total_features: Total number of features.
            rng: Random number generator instance.

        Returns:
            Sorted array of selected feature indices.
        """
        n_select = self._resolve_max_features(n_total_features)
        selected = rng.choice(n_total_features, size=n_select, replace=False)
        return np.sort(selected)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForest":
        """Train the Random Forest on the given data.

        For each tree:
        1. Draw a bootstrap sample from the training data.
        2. Build a DecisionTree that only considers random feature subsets.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self: The trained random forest.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_features = X.shape[1]
        self.trees = []

        base_seed = self.random_state
        for i in range(self.n_estimators):
            # Each tree gets a different random state derived from base seed
            tree_seed = base_seed + i

            # Bootstrap sample
            X_boot, y_boot, oob_indices = self._bootstrap_sample(X, y)

            # Build decision tree with the bootstrap sample
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            # Override random seed for this tree's bootstrap sampling
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  Trained tree {i + 1}/{self.n_estimators}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using majority voting across all trees.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Predicted labels as a 1-D numpy array.
        """
        X = np.asarray(X, dtype=np.float64)

        # Collect predictions from all trees: shape (n_estimators, n_samples)
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        # Majority vote along the tree axis (axis=0)
        predictions = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            counts = np.bincount(all_predictions[:, i])
            predictions[i] = np.argmax(counts)

        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities as the fraction of trees voting for each class.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Probability array of shape (n_samples, n_classes).
        """
        X = np.asarray(X, dtype=np.float64)
        all_predictions = np.array([tree.predict(X) for tree in self.trees])

        n_samples = X.shape[0]
        classes = np.unique(all_predictions)
        n_classes = len(classes)

        proba = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            counts = np.bincount(all_predictions[:, i], minlength=n_classes)
            proba[i] = counts / self.n_estimators

        return proba

    def feature_importance(self) -> np.ndarray:
        """Estimate feature importance based on average impurity decrease.

        Aggregates feature usage statistics across all trees. Since each tree
        may split on different features at various depths, this provides a
        rough importance ranking.

        Returns:
            1-D array of feature importance scores (normalized to sum to 1).
        """
        importance = np.zeros(self.n_features)

        for tree in self.trees:
            importance = self._accumulate_importance(tree.root, importance)

        # Normalize
        total = importance.sum()
        if total > 0:
            importance /= total

        return importance

    def _accumulate_importance(self, node, importance: np.ndarray) -> np.ndarray:
        """Recursively accumulate feature importance from a tree.

        Args:
            node: Current tree node.
            importance: Running importance array.

        Returns:
            Updated importance array.
        """
        if node is None or node.is_leaf():
            return importance

        importance[node.feature_index] += 1
        importance = self._accumulate_importance(node.left, importance)
        importance = self._accumulate_importance(node.right, importance)

        return importance


if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report

    print("=" * 60)
    print("Random Forest Classifier - Demo")
    print("=" * 60)

    # Generate a synthetic classification dataset
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test data:     {X_test.shape[0]} samples, {X_test.shape[1]} features")
    print(f"Classes:       {np.unique(y)}")
    print()

    # Train Random Forest
    rf = RandomForest(
        n_estimators=50,
        max_depth=8,
        max_features="sqrt",
        min_samples_split=5,
        random_state=42,
    )
    print("Training Random Forest...")
    rf.fit(X_train, y_train)
    print("Training complete.\n")

    # Evaluate
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print()

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Feature importance
    importance = rf.feature_importance()
    print("Feature Importances:")
    for i, imp in enumerate(importance):
        bar = "#" * int(imp * 50)
        print(f"  Feature {i:2d}: {imp:.4f} {bar}")

    # Compare with a single decision tree
    print("\n" + "-" * 40)
    print("Comparison: Single Tree vs Random Forest")
    print("-" * 40)

    single_tree = DecisionTree(max_depth=8, min_samples_split=5)
    single_tree.fit(X_train, y_train)
    tree_acc = accuracy_score(y_test, single_tree.predict(X_test))
    print(f"Single Decision Tree Accuracy: {tree_acc:.4f}")
    print(f"Random Forest Accuracy:        {acc:.4f}")
    print(f"Improvement:                   {acc - tree_acc:+.4f}")

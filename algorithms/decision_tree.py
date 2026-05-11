"""
Decision Tree Module - CART Classification Tree from Scratch.

Implements a binary decision tree classifier using the Gini impurity criterion.
Built entirely with NumPy — no external ML library dependencies for the core
algorithm. Supports configurable maximum depth and minimum samples per split.

Example:
    >>> from decision_tree import DecisionTree
    >>> clf = DecisionTree(max_depth=5, min_samples_split=2)
    >>> clf.fit(X_train, y_train)
    >>> preds = clf.predict(X_test)
"""

import numpy as np


class Node:
    """Represents a single node (decision or leaf) in the decision tree.

    Attributes:
        feature_index: Index of the feature used for splitting (None for leaves).
        threshold: Threshold value for the split (None for leaves).
        left: Left child node (samples with feature <= threshold).
        right: Right child node (samples with feature > threshold).
        value: Predicted class label for leaf nodes (None for internal nodes).
    """

    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf(self) -> bool:
        """Return True if this node is a leaf node."""
        return self.value is not None


class DecisionTree:
    """Decision Tree Classifier using the CART algorithm with Gini impurity.

    Supports binary splits on numerical features. The tree is grown greedily
    by selecting the feature and threshold that maximally reduces Gini impurity.

    Attributes:
        max_depth: Maximum depth of the tree.
        min_samples_split: Minimum number of samples required to attempt a split.
        root: Root node of the trained tree.
        n_classes: Number of unique classes in the training data.
    """

    def __init__(self, max_depth: int = 10, min_samples_split: int = 2):
        """Initialize the Decision Tree.

        Args:
            max_depth: Maximum allowed depth of the tree. Use None for unlimited.
            min_samples_split: Minimum number of samples a node must have to
                consider splitting it further.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.n_classes = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTree":
        """Build the decision tree from training data.

        Args:
            X: Training features of shape (n_samples, n_features).
            y: Training labels of shape (n_samples,).

        Returns:
            self: The trained decision tree.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.n_classes = len(np.unique(y))
        self.root = self._build_tree(X, y, depth=0)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for the given samples.

        Args:
            X: Input features of shape (n_samples, n_features).

        Returns:
            Predicted labels as a 1-D numpy array.
        """
        X = np.asarray(X, dtype=np.float64)
        return np.array([self._predict_single(sample, self.root) for sample in X])

    def _predict_single(self, x: np.ndarray, node: Node) -> int:
        """Traverse the tree to predict the label for a single sample.

        Args:
            x: Feature vector of shape (n_features,).
            node: Current tree node.

        Returns:
            Predicted class label.
        """
        if node.is_leaf():
            return node.value

        if x[node.feature_index] <= node.threshold:
            return self._predict_single(x, node.left)
        else:
            return self._predict_single(x, node.right)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively build the decision tree.

        Args:
            X: Feature matrix for the current node's samples.
            y: Labels for the current node's samples.
            depth: Current depth of the tree.

        Returns:
            A Node representing the subtree.
        """
        n_samples = len(y)
        n_classes = np.unique(y)

        # Stopping criteria: pure node, too few samples, or max depth reached
        if (len(n_classes) == 1
                or n_samples < self.min_samples_split
                or (self.max_depth is not None and depth >= self.max_depth)):
            return Node(value=self._leaf_value(y))

        # Find the best split
        best_feature, best_threshold = self._find_best_split(X, y)

        # If no valid split found, create a leaf
        if best_feature is None:
            return Node(value=self._leaf_value(y))

        # Partition the data
        left_idx = X[:, best_feature] <= best_threshold
        right_idx = X[:, best_feature] > best_threshold

        # Safety check: ensure both partitions have samples
        if np.sum(left_idx) == 0 or np.sum(right_idx) == 0:
            return Node(value=self._leaf_value(y))

        left_subtree = self._build_tree(X[left_idx], y[left_idx], depth + 1)
        right_subtree = self._build_tree(X[right_idx], y[right_idx], depth + 1)

        return Node(
            feature_index=best_feature,
            threshold=best_threshold,
            left=left_subtree,
            right=right_subtree,
        )

    def _find_best_split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find the best feature and threshold to split on.

        Evaluates all unique thresholds for each feature and selects the split
        that yields the greatest reduction in Gini impurity.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,).

        Returns:
            A tuple (best_feature_index, best_threshold). If no split improves
            the impurity, returns (None, None).
        """
        n_samples, n_features = X.shape
        best_gain = -1.0
        best_feature = None
        best_threshold = None
        parent_gini = self._calculate_gini(y)

        for feature_idx in range(n_features):
            # Get unique sorted thresholds for this feature
            thresholds = np.unique(X[:, feature_idx])

            # For efficiency, sample at most 50 thresholds for continuous features
            if len(thresholds) > 50:
                thresholds = np.percentile(thresholds, np.linspace(0, 100, 50))

            for threshold in thresholds:
                left_mask = X[:, feature_idx] <= threshold
                right_mask = ~left_mask

                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                # Weighted Gini of children
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                gini_left = self._calculate_gini(y[left_mask])
                gini_right = self._calculate_gini(y[right_mask])
                weighted_gini = (n_left * gini_left + n_right * gini_right) / n_samples

                # Information gain = parent_gini - weighted_child_gini
                gain = parent_gini - weighted_gini

                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold

    @staticmethod
    def _calculate_gini(y: np.ndarray) -> float:
        """Calculate the Gini impurity for a set of labels.

        Gini = 1 - sum(p_k^2) for each class k, where p_k is the proportion
        of class k in the node.

        Args:
            y: Labels of shape (n_samples,).

        Returns:
            Gini impurity value in [0, 0.5] for binary classification,
            [0, 1 - 1/K] for K classes.
        """
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1.0 - np.sum(probabilities ** 2)

    @staticmethod
    def _leaf_value(y: np.ndarray) -> int:
        """Determine the predicted class for a leaf node.

        Uses majority voting: the class with the highest count wins.

        Args:
            y: Labels of shape (n_samples,).

        Returns:
            The majority class label.
        """
        classes, counts = np.unique(y, return_counts=True)
        return classes[np.argmax(counts)]

    def print_tree(self, node: Node = None, indent: str = ""):
        """Print a text representation of the decision tree.

        Args:
            node: Current node (defaults to root).
            indent: Indentation string for recursive printing.
        """
        if node is None:
            node = self.root

        if node.is_leaf():
            print(f"{indent}Leaf -> class {node.value}")
            return

        print(f"{indent}[X{node.feature_index} <= {node.threshold:.4f}]")
        print(f"{indent}  Left:")
        self.print_tree(node.left, indent + "    ")
        print(f"{indent}  Right:")
        self.print_tree(node.right, indent + "    ")


if __name__ == "__main__":
    print("=" * 60)
    print("Decision Tree Classifier - Demo")
    print("=" * 60)

    # Generate a synthetic classification dataset (moons-like)
    np.random.seed(42)
    n_samples = 300
    X_class0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
    X_class1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
    X = np.vstack([X_class0, X_class1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train the decision tree
    tree = DecisionTree(max_depth=5, min_samples_split=5)
    tree.fit(X_train, y_train)

    # Evaluate
    y_pred = tree.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # Show some predictions
    print("\nSample predictions:")
    for i in range(min(10, len(X_test))):
        print(f"  X={X_test[i]} -> True: {y_test[i]}, Pred: {y_pred[i]}")

    # Print tree structure (truncated for readability)
    print("\nTree Structure (first few levels):")
    tree.print_tree(tree.root)

    # Demo: Iris-like multi-class dataset
    print("\n" + "=" * 60)
    print("Decision Tree - Multi-class Demo")
    print("=" * 60)

    np.random.seed(123)
    # Three clusters
    X3 = np.vstack([
        np.random.randn(50, 2) + np.array([0, 0]),
        np.random.randn(50, 2) + np.array([3, 3]),
        np.random.randn(50, 2) + np.array([-3, 3]),
    ])
    y3 = np.array([0] * 50 + [1] * 50 + [2] * 50)

    X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y3, test_size=0.3, random_state=0)

    tree3 = DecisionTree(max_depth=8, min_samples_split=2)
    tree3.fit(X3_train, y3_train)
    y3_pred = tree3.predict(X3_test)
    acc3 = np.mean(y3_pred == y3_test)
    print(f"Multi-class Test Accuracy: {acc3:.4f}")

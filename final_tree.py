from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Node:
    """
    A node in a decision tree.

    Parameters
    ----------
    feature : int, optional (default=None)
        The feature index used for splitting the node.
    threshold : float, optional (default=None)
        The threshold value at the node.
    n_samples : int, optional (default=None)
        The number of samples at the node.
    value : int, optional (default=None)
        The value of the node (i.e., the mean target value of the samples at the node).
    mse : float, optional (default=None)
        The mean squared error of the node (i.e., the impurity criterion).
    left : Node, optional (default=None)
        The left child node.
    right : Node, optional (default=None)
        The right child node.
    """

    feature: int = None
    threshold: float = None
    n_samples: int = None
    value: int = None
    mse: float = None
    left: Node = None
    right: Node = None


@dataclass
class DecisionTreeRegressor:
    """
    Decision tree regressor.

    Parameters
    ----------
    max_depth : int, optional (default=None)
        The maximum depth of the tree. If None, then nodes are expanded until
        all leaves are pure or until all leaves contain less than
        min_samples_split samples.
    min_samples_split : int, optional (default=2)
        The minimum number of samples required to split an internal node.

    Attributes
    ----------
    tree_ : Node
        The root node of the decision tree.
    n_features_ : int
        The number of features when fit is performed.
    """

    max_depth: int
    min_samples_split: int = 2

    def fit(self, X: np.ndarray, y: np.ndarray) -> DecisionTreeRegressor:
        """
        Build a decision tree regressor from the training set (X, y).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array-like, shape (n_samples,)
            The target values (real numbers in regression).

        Returns
        -------
        self : object
            Returns self.
        """
        self.n_features_ = X.shape[1]
        self.tree_ = self._split_node(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict regression target for X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : array of shape (n_samples,)
            The predicted values.
        """
        return [self._predict_one_sample(features) for features in X]

    def as_json(self) -> str:
        """Return the decision tree as a JSON string."""
        return self._as_json(self.tree_) or ""

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse impurity criterion for a given set of target values."""
        return np.mean((y - np.mean(y)) ** 2)

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mse criterion for a given set of target values."""
        num = self._mse(y_left) * y_left.size + self._mse(y_right) * y_right.size
        den = y_left.size + y_right.size
        return num / den

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
        """Find the best split for a node."""
        node_size = y.size
        if node_size < self.min_samples_split:
            return None, None

        node_mse = self._mse(y)
        best_mse = node_mse
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            thresholds = np.unique(X[:, idx])
            for thr in thresholds:
                left = y[X[:, idx] <= thr]
                right = y[X[:, idx] > thr]

                if left.size == 0 or right.size == 0:
                    continue

                weihted_mse = self._weighted_mse(left, right)
                if weihted_mse < best_mse:
                    best_mse = weihted_mse
                    best_idx = idx
                    best_thr = thr

        return best_idx, best_thr

    def _split_node(self, X: np.ndarray, y: np.ndarray, depth: int = 0) -> Node:
        """Split a node and return the resulting left and right child nodes."""
        value = int(round(np.mean(y)))
        node = Node(value=value, mse=self._mse(y), n_samples=len(y))

        if not self.max_depth or depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                left_idx = X[:, idx] <= thr
                X_left, y_left = X[left_idx], y[left_idx]
                X_right, y_right = X[~left_idx], y[~left_idx]
                node.feature = idx
                node.threshold = thr
                node.left = self._split_node(X_left, y_left, depth + 1)
                node.right = self._split_node(X_right, y_right, depth + 1)

        return node

    def _predict_one_sample(self, features: np.ndarray) -> int:
        """Predict the target value of a single sample."""
        node = self.tree_
        while node.left:
            if features[node.feature] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def _as_json(self, node: Node) -> str:
        """Return the decision tree as a JSON string."""
        if node is None:
            return None

        mse = round(node.mse, 2) if node.mse is not None else None
        if node.left is None and node.right is None:
            return (
                f'{{"value": {node.value}, '
                f'"n_samples": {node.n_samples}, "mse": {mse}}}'
            )
        else:
            return (
                f'{{"feature": {node.feature}, "threshold": {node.threshold}, '
                f'"n_samples": {node.n_samples}, "mse": {mse}, '
                f'"left": {self._as_json(node.left)}, '
                f'"right": {self._as_json(node.right)}}}'
            )


def run():
    """Run the example."""

    DEPTH = 3
    PRINT_TREE = True

    df = pd.read_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "sample.csv")
    )

    df = df.reset_index(drop=True)
    data = df.drop("delay_days", axis=1)
    target = df["delay_days"]

    # Split the dataset into training and test sets
    np.random.seed(42)
    idx = data.index.values
    np.random.shuffle(idx)
    train_idx = idx[: int(0.8 * len(idx))]
    test_idx = idx[int(0.8 * len(idx)) :]

    X_train, X_test = data.iloc[train_idx], data.iloc[test_idx]
    y_train, y_test = target.iloc[train_idx], target.iloc[test_idx]

    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

    # Train the decision tree regressor
    tree = DecisionTreeRegressor(max_depth=DEPTH)
    tree.fit(X_train, y_train)

    # Predict the targets of the test set
    y_pred = tree.predict(X_test)

    # Evaluate the performance of the decision tree regressor
    rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
    print("RMSE:", round(rmse, 2))

    # Print the decision tree
    if PRINT_TREE:
        print(tree.as_json())


if __name__ == "__main__":
    run()

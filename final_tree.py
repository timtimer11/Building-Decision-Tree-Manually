from dataclasses import dataclass
from typing import Tuple

import numpy as np


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
    left: "Node" = None
    right: "Node" = None


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

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeRegressor":
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

    def _mse(self, y: np.ndarray) -> float:
        """Compute the mse impurity criterion for a given set of target values."""
        return np.mean((y - np.mean(y)) ** 2)

    def _weighted_mse(self, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute the weighted mse criterion for a given set of target values."""
        num = self._mse(y_left) * y_left.size + self._mse(y_right) * y_right.size
        den = y_left.size + y_right.size
        return num / den

    def _best_split(self, X: np.ndarray, y: np.ndarray) -> Tuple[int, float]:
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

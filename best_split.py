from __future__ import annotations

import numpy as np


def mse(y: np.ndarray) -> float:
    """Calculate the MSE of a vector"""
    mse = np.mean((y - np.mean(y)) ** 2)
    return mse


def weighted_mse(y_left: np.ndarray, y_right: np.ndarray) -> float:
    """Calculate the weighted MSE of two vectors."""

    mse_left = mse(y_left)
    mse_right = mse(y_right)

    N_left = y_left.size
    N_right = y_right.size

    mse_weighted = ((mse_left * N_left) + (mse_right * N_right)) / (N_left + N_right)

    return mse_weighted


def best_split(X: np.ndarray, y: np.ndarray) -> tuple[int, float]:
    """Find the best split for a node"""
    best_mse = mse(y)
    best_threshold = None
    best_feature = None

    for feature in range(X.shape[1]):
        thresholds = np.unique(X[:, feature])
        for threshold in thresholds:
            left_mask = X[:, feature] <= threshold
            right_mask = X[:, feature] > threshold

            y_left = y[left_mask]
            y_right = y[right_mask]

            if y_left.size == 0 or y_right.size == 0:
                continue
            mse_weighted = weighted_mse(y_left, y_right)

            if mse_weighted < best_mse:
                best_mse = mse_weighted
                best_threshold = threshold
                best_feature = feature

    return best_feature, best_threshold

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


def split(X: np.ndarray, y: np.ndarray, feature: int) -> float:
    """Find the best split for a node (one feature)"""
    # iterate through all values of the feature

    thresholds = np.unique(X[:, feature])
    best_threshold = None
    best_mse = mse(y)

    if y.size < 2:
        return best_threshold

    for threshold in thresholds:
        # split the data
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold

        y_left = y[left_mask]
        y_right = y[right_mask]

        if y_left.size == 0 or y_right.size == 0:
            continue

        # calculate the weighted mse
        mse_weighted = (weighted_mse(y_left, y_right))

        # update best threshold
        if mse_weighted < best_mse:
            best_mse = mse_weighted
            best_threshold = threshold

    return best_threshold

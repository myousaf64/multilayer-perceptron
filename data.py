"""Load / preprocess the Wisconsin breast-cancer CSV (no header).

Column 0 = id, column 1 = diagnosis (M/B), columns 2..31 = 30 features.
"""
import numpy as np

N_FEATURES = 30


def load_raw(path):
    rows = np.genfromtxt(path, delimiter=',', dtype=str)
    ids = rows[:, 0]
    labels = rows[:, 1]
    X = rows[:, 2:].astype(float)
    return ids, labels, X


def one_hot(labels):
    """M -> [1,0] (malignant, positive class), B -> [0,1]."""
    Y = np.zeros((len(labels), 2))
    Y[labels == 'M', 0] = 1
    Y[labels == 'B', 1] = 1
    return Y


def standardize(X, mean=None, std=None):
    if mean is None:
        mean = X.mean(axis=0)
        std = X.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X - mean) / std, mean, std

#!/usr/bin/env python3
"""predict.py [data_valid.csv] [saved_model.npy] — predict and score with binary cross-entropy."""
import argparse
import numpy as np
import data
from mlp import MLP


def binary_cross_entropy(y, p):
    """y = 1 for malignant; p = predicted P(malignant). E = -mean(y ln p + (1-y) ln(1-p))."""
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', nargs='?', default='data_valid.csv')
    ap.add_argument('model', nargs='?', default='saved_model.npy')
    args = ap.parse_args()

    m = np.load(args.model, allow_pickle=True).item()
    net = MLP(m['sizes'])
    net.W, net.b = m['W'], m['b']

    _, labels, X = data.load_raw(args.dataset)
    X, _, _ = data.standardize(X, m['mean'], m['std'])
    proba = net.predict_proba(X)          # column 0 = P(malignant)

    y = (labels == 'M').astype(float)
    p_malignant = proba[:, 0]
    acc = float(np.mean((p_malignant >= 0.5) == (y == 1)))
    print(f'binary cross-entropy loss: {binary_cross_entropy(y, p_malignant):.4f}')
    print(f'accuracy: {acc:.4f} on {len(y)} examples')


if __name__ == '__main__':
    main()

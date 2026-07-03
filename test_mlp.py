"""Self-check for the MLP. Run: python3 test_mlp.py

1. Numerical gradient check — finite differences vs analytic backprop gradients.
   This is the load-bearing test: if backprop is wrong, it fails here.
2. End-to-end — train on a split of data.csv and assert good validation accuracy.
"""
import numpy as np
import data
from mlp import MLP


def gradient_check():
    rng = np.random.default_rng(0)
    net = MLP([4, 5, 3, 2], seed=1)
    X = rng.normal(size=(6, 4))
    Y = np.zeros((6, 2))
    Y[np.arange(6), rng.integers(0, 2, 6)] = 1

    dW, db = net.gradients(X, Y)
    eps = 1e-5
    for li in range(len(net.W)):
        for _ in range(20):                     # spot-check random entries
            i = rng.integers(net.W[li].shape[0])
            j = rng.integers(net.W[li].shape[1])
            orig = net.W[li][i, j]
            net.W[li][i, j] = orig + eps
            lp = net.cce(Y, net.predict_proba(X))
            net.W[li][i, j] = orig - eps
            lm = net.cce(Y, net.predict_proba(X))
            net.W[li][i, j] = orig
            num = (lp - lm) / (2 * eps)
            assert abs(num - dW[li][i, j]) < 1e-6, f'grad mismatch W[{li}][{i},{j}]: {num} vs {dW[li][i, j]}'
    print('gradient check: analytic == numerical (backprop correct)')


def end_to_end():
    _, labels, X = data.load_raw('data.csv')
    rng = np.random.default_rng(42)
    order = rng.permutation(len(X))
    cut = int(len(X) * 0.8)
    tr, va = order[:cut], order[cut:]
    Xtr, mean, std = data.standardize(X[tr])
    Xva, _, _ = data.standardize(X[va], mean, std)
    Ytr, Yva = data.one_hot(labels[tr]), data.one_hot(labels[va])

    net = MLP([data.N_FEATURES, 24, 24, 2], seed=42)
    h = net.fit(Xtr, Ytr, Xva, Yva, epochs=84, lr=0.0314, batch_size=8, verbose=False)
    val_acc = h['val_acc'][-1]
    print(f'final val_loss={h["val_loss"][-1]:.4f} val_acc={val_acc:.4f}')
    assert h['val_loss'][-1] < h['val_loss'][0], 'validation loss did not decrease'
    assert val_acc >= 0.95, f'val accuracy {val_acc:.4f} below 0.95'


if __name__ == '__main__':
    gradient_check()
    end_to_end()
    print('OK: MLP checks pass')

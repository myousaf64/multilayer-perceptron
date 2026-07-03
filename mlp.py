"""Multilayer perceptron from scratch (numpy for linear algebra only).

Dense layers, sigmoid hidden activations, softmax output, categorical
cross-entropy, mini-batch gradient descent with backpropagation. No neural-network
library is used — feedforward, backprop and the update rule are all hand-written.
"""
import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def he_uniform(fan_in, fan_out, rng):
    limit = np.sqrt(6.0 / fan_in)
    return rng.uniform(-limit, limit, size=(fan_in, fan_out))


class MLP:
    def __init__(self, sizes, seed=42):
        """sizes = [n_input, hidden..., n_output]; >=2 hidden layers expected."""
        self.sizes = sizes
        rng = np.random.default_rng(seed)
        self.W = [he_uniform(sizes[i], sizes[i + 1], rng) for i in range(len(sizes) - 1)]
        self.b = [np.zeros((1, sizes[i + 1])) for i in range(len(sizes) - 1)]

    def forward(self, X):
        activations = [X]
        a = X
        for i in range(len(self.W)):
            z = a @ self.W[i] + self.b[i]
            a = softmax(z) if i == len(self.W) - 1 else sigmoid(z)
            activations.append(a)
        return activations

    def predict_proba(self, X):
        return self.forward(X)[-1]

    @staticmethod
    def cce(Y, P):
        return float(-np.mean(np.sum(Y * np.log(np.clip(P, 1e-12, 1.0)), axis=1)))

    @staticmethod
    def accuracy(Y, P):
        return float(np.mean(P.argmax(1) == Y.argmax(1)))

    def gradients(self, Xb, Yb):
        acts = self.forward(Xb)
        m = Xb.shape[0]
        dW = [None] * len(self.W)
        db = [None] * len(self.b)
        dZ = acts[-1] - Yb                      # softmax + CCE gradient
        for i in reversed(range(len(self.W))):
            dW[i] = acts[i].T @ dZ / m
            db[i] = dZ.mean(axis=0, keepdims=True)
            if i > 0:
                dA = dZ @ self.W[i].T
                dZ = dA * acts[i] * (1 - acts[i])   # sigmoid derivative
        return dW, db

    def _step(self, Xb, Yb, lr):
        dW, db = self.gradients(Xb, Yb)
        for i in range(len(self.W)):
            self.W[i] -= lr * dW[i]
            self.b[i] -= lr * db[i]

    def fit(self, Xtr, Ytr, Xval, Yval, epochs=84, lr=0.0314, batch_size=8,
            seed=42, verbose=True):
        rng = np.random.default_rng(seed)
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}
        n = Xtr.shape[0]
        for ep in range(1, epochs + 1):
            order = rng.permutation(n)
            for s in range(0, n, batch_size):
                idx = order[s:s + batch_size]
                self._step(Xtr[idx], Ytr[idx], lr)
            Ptr, Pval = self.predict_proba(Xtr), self.predict_proba(Xval)
            history['loss'].append(self.cce(Ytr, Ptr))
            history['val_loss'].append(self.cce(Yval, Pval))
            history['acc'].append(self.accuracy(Ytr, Ptr))
            history['val_acc'].append(self.accuracy(Yval, Pval))
            if verbose:
                print(f'epoch {ep:02d}/{epochs} - loss: {history["loss"][-1]:.4f} '
                      f'- val_loss: {history["val_loss"][-1]:.4f}')
        return history

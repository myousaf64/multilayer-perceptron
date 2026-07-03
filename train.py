#!/usr/bin/env python3
"""train.py [data_train.csv] [data_valid.csv] — train the MLP and save the model.

Prints per-epoch loss/val_loss, plots the loss and accuracy learning curves, and
saves topology + weights + feature scaler to saved_model.npy.
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import data
from mlp import MLP

MODEL_FILE = 'saved_model.npy'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('train_csv', nargs='?', default='data_train.csv')
    p.add_argument('valid_csv', nargs='?', default='data_valid.csv')
    p.add_argument('--layer', type=int, nargs='+', default=[24, 24])
    p.add_argument('--epochs', type=int, default=84)
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--learning_rate', type=float, default=0.0314)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    _, ytr, Xtr = data.load_raw(args.train_csv)
    _, yval, Xval = data.load_raw(args.valid_csv)
    Xtr, mean, std = data.standardize(Xtr)
    Xval, _, _ = data.standardize(Xval, mean, std)
    Ytr, Yval = data.one_hot(ytr), data.one_hot(yval)

    print(f'x_train shape : {Xtr.shape}')
    print(f'x_valid shape : {Xval.shape}')

    sizes = [data.N_FEATURES] + args.layer + [2]
    net = MLP(sizes, seed=args.seed)
    history = net.fit(Xtr, Ytr, Xval, Yval, epochs=args.epochs,
                      lr=args.learning_rate, batch_size=args.batch_size, seed=args.seed)

    np.save(MODEL_FILE, {'sizes': sizes, 'W': net.W, 'b': net.b,
                         'mean': mean, 'std': std}, allow_pickle=True)
    print(f"> saving model '{MODEL_FILE}' to disk...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(history['loss'], label='training loss')
    ax1.plot(history['val_loss'], label='validation loss')
    ax1.set_title('Loss'); ax1.set_xlabel('epoch'); ax1.legend()
    ax2.plot(history['acc'], label='training acc')
    ax2.plot(history['val_acc'], label='validation acc')
    ax2.set_title('Accuracy'); ax2.set_xlabel('epoch'); ax2.legend()
    fig.tight_layout()
    fig.savefig('learning_curves.png', dpi=100)
    print('saved learning_curves.png')


if __name__ == '__main__':
    main()

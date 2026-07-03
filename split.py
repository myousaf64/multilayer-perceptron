#!/usr/bin/env python3
"""split.py [data.csv] [--ratio 0.8] [--seed 42] — split into train/valid CSVs."""
import sys
import argparse
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('dataset', nargs='?', default='data.csv')
    p.add_argument('--ratio', type=float, default=0.8)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    rows = np.genfromtxt(args.dataset, delimiter=',', dtype=str)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(rows))
    cut = int(len(rows) * args.ratio)
    train, valid = rows[order[:cut]], rows[order[cut:]]

    np.savetxt('data_train.csv', train, delimiter=',', fmt='%s')
    np.savetxt('data_valid.csv', valid, delimiter=',', fmt='%s')
    print(f'train: {len(train)} rows -> data_train.csv')
    print(f'valid: {len(valid)} rows -> data_valid.csv')


if __name__ == '__main__':
    main()

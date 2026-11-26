# multilayer-perceptron

[![CI](https://github.com/myousaf64/multilayer-perceptron/actions/workflows/ci.yml/badge.svg)](https://github.com/myousaf64/multilayer-perceptron/actions/workflows/ci.yml)

## What it is

A feedforward neural network with backpropagation, written from scratch, trained to
classify breast-cancer diagnoses (malignant vs benign) from 30 features. Validation
accuracy is about 99%, and backpropagation is verified against a numerical gradient. A 42
Abu Dhabi project.

## Demo

<!-- TODO: screenshot or GIF -->
![Learning curves](docs/demo.png)

## Stack

- Python with numpy for linear algebra only. The network mathematics is hand-written: no
  neural-network library.
- matplotlib to plot the learning curves.
- `mlp.py` holds the layers and training; `data.py` loads and standardises the CSV.

## Quickstart

```
python3 split.py data.csv --ratio 0.8 --seed 42
python3 train.py data_train.csv data_valid.csv --layer 24 24 --epochs 84 \
        --batch_size 8 --learning_rate 0.0314
python3 predict.py data_valid.csv saved_model.npy
```

`train.py` prints per-epoch loss and val_loss, saves `saved_model.npy` (topology, weights
and the standardisation scaler), and plots `learning_curves.png`. `predict.py` reloads the
model and reports the subject's binary cross-entropy plus accuracy.

## Results

- Validation accuracy about 99%.
- Dense layers, sigmoid hidden activation, softmax output, categorical cross-entropy loss,
  heUniform initialisation, two hidden layers (`[24, 24]`) by default.

Verify it:

```
python3 test_mlp.py
```

This runs a finite-difference gradient check against the analytic gradient (proving
backpropagation is correct), then an end-to-end run that asserts validation accuracy of at
least 0.95. See [`PROGRESS.md`](PROGRESS.md) for the development log.

## License

Apache-2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

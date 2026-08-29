# multilayer-perceptron

A feedforward neural network with backpropagation, written from scratch, trained
to classify breast-cancer diagnoses. Validation accuracy is about 99%, and
backpropagation is verified against a numerical gradient. A 42 Abu Dhabi project.

## Run

```
python3 split.py data.csv --ratio 0.8 --seed 42
python3 train.py data_train.csv data_valid.csv --layer 24 24 --epochs 84 \
        --batch_size 8 --learning_rate 0.0314
python3 predict.py data_valid.csv saved_model.npy
```

`train.py` saves `saved_model.npy` and plots `learning_curves.png`.

## Test

```
python3 test_mlp.py
```

Runs a finite-difference gradient check against the analytic gradient, then an
end-to-end run asserting validation accuracy of at least 0.95.

## Notes

- Dense layers, sigmoid hidden activation, softmax output, categorical
  cross-entropy loss, heUniform initialisation, two hidden layers by default.
- `predict.py` reports the binary cross-entropy the subject asks for.
- numpy is used for linear algebra only. The network mathematics is hand-written.
- `PROGRESS.md` is the development log.

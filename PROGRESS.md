# multilayer-perceptron — progress

**Status: mandatory DONE & verified.** Backprop numerically verified; val acc ~99%.

## Programs (the three required)
- `split.py [data.csv] [--ratio --seed]` — writes `data_train.csv` / `data_valid.csv`.
- `train.py [train] [valid] [--layer 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314]`
  — feedforward + backprop + mini-batch GD; prints per-epoch loss/val_loss; saves
  `saved_model.npy` (topology + weights + scaler); plots `learning_curves.png`
  (loss + accuracy).
- `predict.py [data_valid.csv] [saved_model.npy]` — loads model, evaluates with the
  subject's **binary cross-entropy** and prints accuracy.

## Design
- `mlp.py` — Dense layers, **sigmoid** hidden, **softmax** output, **categorical
  cross-entropy**, **heUniform** init. Default ≥2 hidden layers ([24,24]).
- `data.py` — load raw CSV (id, M/B label, 30 features), one-hot (M=[1,0]),
  standardise with train mean/std (reused at predict time from the saved model).
- numpy for linear algebra only; the network math is all hand-written (no NN lib).

## Verification
`python3 test_mlp.py`:
- **numerical gradient check** (finite diff vs analytic) — proves backprop correct;
- end-to-end: val_loss decreases, val accuracy ≥0.95 (actual ~0.99).

## Next (bonus, only if mandatory perfect)
- Adam / momentum / RMSprop; early stopping; metrics history; overlaid curves.

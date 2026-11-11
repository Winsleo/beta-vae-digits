# beta-VAE on binarised digits (AI Engineering, homework 1)

Code for the P1 experiment (beta-VAE with a 2-d latent on 8x8 digits) and for the small
demos behind Q2 (target leakage) and Q4 (autoencoder vs VAE).

## Layout

```
src/
  config.py          constants of the protocol (betas, seeds, batch size, ...)
  data.py            binarised digits + fixed stratified 60/20/20 split
  model.py           VAE, KL, Bernoulli NLL, beta-VAE loss
  train.py           train all betas, save models, history and epoch snapshots
  evaluate.py        test metrics: recon NLL, KL, -ELBO, IWAE, posterior and prior-sample statistics
  plots.py           all figures (reconstruction, prior samples, scatter, interpolation, manifold, curves)
  multiseed.py       repeat training with several seeds
  q2_leakage.py      Q2 demo: leaky vs masked autoregressive model
  q4_autoencoder.py  Q4 demo: deterministic AE sampled from N(0, I)
tests/test_core.py   acceptance checks (KL, gradients through the reparameterisation, split, IWAE)
Makefile             make test | train | evaluate | plots | extras | multiseed | all
```

## Setup

Python 3.9+ with `torch`, `numpy`, `scikit-learn`, `matplotlib` (`pip install -r requirements.txt`).
CPU is enough; the whole pipeline (`make all`) takes a few minutes.

## Run

```bash
make test        # 8 checks
make train       # 4 betas x 500 epochs, snapshot at epoch 100  -> runs/main/
make evaluate    # results/main.json, results/ep100.json
make plots       # results/figs/*.png
make extras      # results/q2, results/q4
make multiseed   # results/multiseed.json (seeds 0-4)
```

or call the scripts directly, e.g. `python src/train.py --epochs 100 --out runs/short`.
Every script has `--help`. `runs/` and `results/` are not tracked by git.

## Protocol

| | |
|---|---|
| data | `load_digits()`, pixels / 16, binarised with `>= 0.5`, flattened to 64 values |
| split | stratified 60 / 20 / 20 (1078 / 359 / 360), `random_state=42` for both `train_test_split` calls |
| model | encoder 64 -> 128 (ReLU) -> (mu, logvar), latent 2; decoder 2 -> 128 (ReLU) -> 64 logits |
| loss | mean over the batch of `BCE_sum + beta * KL`, beta in {0, 0.1, 1, 4}; beta = 0 still samples |
| training | Adam, lr 1e-3, batch 128, seed 42 (init and per-epoch shuffle), last epoch is the model |
| validation | monitored every epoch with one posterior sample, never used to choose a model |
| evaluation | test set; recon NLL = mean of 20 posterior samples (seed 12345); KL closed form; -ELBO = recon + KL |

Betas are compared on the standard negative ELBO (recon NLL + KL), not on the weighted training loss.

## Reproduced numbers (seed 42, test set)

500 epochs:

| beta | recon NLL | KL | -ELBO | IWAE-1000 |
|---|---:|---:|---:|---:|
| 0 | 16.9926 | 97.6064 | 114.5990 | 113.32 |
| 0.1 | 16.3120 | 7.6766 | 23.9886 | 22.40 |
| 1 | 16.9103 | 3.2577 | 20.1680 | 19.51 |
| 4 | 23.4790 | 0.4105 | 23.8895 | 22.16 |

100 epochs (snapshot of the same run): -ELBO 65.2444 / 23.7576 / 20.8564 / 24.4683.
Independent-Bernoulli baseline (no latent): 25.28.

Five more seeds (0-4, 500 epochs), mean +- std of -ELBO: 113.69 +- 10.44, 24.08 +- 0.42, 20.24 +- 0.12, 24.02 +- 0.04.

These match the numbers in the report. They were reproduced with torch 2.6.0, numpy 1.26.4 and
scikit-learn 1.7.2; the original runs used torch 2.7.1. The metrics that do not involve the
digit classifier agree to the printed precision. The class-entropy columns can differ in the
last digit (about 0.01) between scikit-learn versions.

## Notes

- The model layers are created in a fixed order (see `model.py`): it determines the initial weights, hence the numbers.
- The split is computed on the fly and is deterministic, so no index files are stored.
- `src/q2_leakage.py` seeds before building each model. A first version of this demo initialised the models before seeding,
  so its numbers moved slightly from run to run.

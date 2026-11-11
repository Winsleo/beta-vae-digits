"""Repeat training with several seeds (init and shuffling change; the data split does not).

    python src/multiseed.py --seeds 0 1 2 3 4 --epochs 500 --out results/multiseed.json
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from config import BETAS, EPOCHS, beta_label
from data import load_splits
from evaluate import prior_sample_stats, reconstruction_metrics
from train import train_one


def _job(args):
    seed, beta, epochs = args
    torch.set_num_threads(1)
    s = load_splits()
    model, _, _ = train_one(beta, s.x_train, s.x_val, epochs, seed)
    model.eval()
    clf = LogisticRegression(max_iter=2000).fit(s.x_train, s.y_train)
    r = reconstruction_metrics(model, torch.from_numpy(s.x_test))
    r.update(prior_sample_stats(model, clf))
    return {"seed": seed, "beta": beta, **{k: r[k] for k in ("recon_nll", "kl", "neg_elbo", "class_entropy")}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("results/multiseed.json"))
    a = ap.parse_args()
    with Pool(a.jobs) as pool:
        rows = pool.map(_job, [(s, b, a.epochs) for s in a.seeds for b in BETAS], chunksize=1)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rows, indent=1))
    print(f"mean +- std over seeds {a.seeds} (n={len(a.seeds)})")
    print("| beta | recon NLL | KL | -ELBO | prior class entropy |\n|---|---:|---:|---:|---:|")
    for b in BETAS:
        rs = [r for r in rows if r["beta"] == b]
        cell = lambda k: f"{np.mean([r[k] for r in rs]):.2f} +- {np.std([r[k] for r in rs], ddof=1):.2f}"
        print(f"| {beta_label(b)} | {cell('recon_nll')} | {cell('kl')} | {cell('neg_elbo')} | {cell('class_entropy')} |")


if __name__ == "__main__":
    main()

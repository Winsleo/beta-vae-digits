"""Train the beta-VAE for every beta in config.BETAS.

    python src/train.py --epochs 500 --out runs/main --snapshots 100

Writes <out>/models/<beta>.pt, <out>/history/<beta>.json and, for each epoch in
--snapshots, <out>/models_ep<epoch>/<beta>.pt. The last-epoch model is the one
that gets evaluated; the validation set is only monitored, never used to pick a model.
"""
import argparse
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch

from config import BATCH_SIZE, BETAS, EPOCHS, EVAL_SEED, LR, TRAIN_SEED, beta_label, beta_tag
from data import load_splits
from model import VAE, beta_vae_loss, bce_sum, kl_divergence


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def monitor(model: VAE, x_val: torch.Tensor):
    """One posterior sample on the validation set (monitoring only)."""
    model.eval()
    mu, logvar = model.encode(x_val)
    gen = torch.Generator().manual_seed(EVAL_SEED)
    z = mu + torch.exp(0.5 * logvar) * torch.randn(mu.shape, generator=gen)
    return float(bce_sum(model.decode(z), x_val).mean()), float(kl_divergence(mu, logvar).mean())


def train_one(beta, x_train, x_val, epochs=EPOCHS, seed=TRAIN_SEED, snapshots=()):
    """Train one model. Returns (model, history, {epoch: state_dict} for the requested snapshots)."""
    set_seed(seed)
    model = VAE()
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    xt, xv = torch.from_numpy(x_train), torch.from_numpy(x_val)
    history, snaps = [], {}
    for epoch in range(1, epochs + 1):
        model.train()
        gen = torch.Generator().manual_seed(seed + epoch)
        perm = torch.randperm(len(xt), generator=gen)
        bce_tot = kl_tot = 0.0
        for start in range(0, len(xt), BATCH_SIZE):
            xb = xt[perm[start:start + BATCH_SIZE]]
            opt.zero_grad(set_to_none=True)
            logits, mu, logvar, _ = model(xb)
            loss, bce, kl = beta_vae_loss(logits, xb, mu, logvar, beta)
            loss.backward()
            opt.step()
            bce_tot += float(bce.detach().sum())
            kl_tot += float(kl.detach().sum())
        val_bce, val_kl = monitor(model, xv)
        history.append({"epoch": epoch, "train_bce": bce_tot / len(xt), "train_kl": kl_tot / len(xt),
                        "val_bce": val_bce, "val_kl": val_kl})
        if epoch in snapshots:
            snaps[epoch] = {k: v.clone() for k, v in model.state_dict().items()}
    return model, history, snaps


def _job(args):
    beta, epochs, seed, snapshots, out = args
    torch.set_num_threads(1)
    s = load_splits()
    model, history, snaps = train_one(beta, s.x_train, s.x_val, epochs, seed, snapshots)
    tag = beta_tag(beta)
    for sub in ("models", "history"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "models" / f"{tag}.pt")
    (out / "history" / f"{tag}.json").write_text(json.dumps(history))
    for epoch, state in snaps.items():
        d = out / f"models_ep{epoch}"
        d.mkdir(parents=True, exist_ok=True)
        torch.save(state, d / f"{tag}.pt")
    last = history[-1]
    return f"beta={beta_label(beta)}  epochs={epochs}  val_bce={last['val_bce']:.3f}  val_kl={last['val_kl']:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--seed", type=int, default=TRAIN_SEED)
    ap.add_argument("--out", type=Path, default=Path("runs/main"))
    ap.add_argument("--betas", type=float, nargs="+", default=list(BETAS))
    ap.add_argument("--snapshots", type=int, nargs="*", default=[], help="extra epochs to save (e.g. 100)")
    ap.add_argument("--jobs", type=int, default=len(BETAS), help="parallel processes")
    a = ap.parse_args()
    jobs = [(b, a.epochs, a.seed, tuple(a.snapshots), a.out) for b in a.betas]
    with Pool(min(a.jobs, len(jobs))) as pool:
        for line in pool.imap(_job, jobs):
            print(line, flush=True)
    print(f"saved to {a.out}/")


if __name__ == "__main__":
    main()

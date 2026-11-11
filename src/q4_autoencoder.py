"""Q4 demo: a deterministic autoencoder (2-d code) sampled with z ~ N(0, I).

    python src/q4_autoencoder.py --out results/q4

Same layers as the VAE but no sampling and no KL term. Reports reconstruction NLL, the scale of
the learned codes, and how diverse the decoded prior samples are (class entropy).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression

from config import EVAL_SEED, N_PRIOR
from data import load_splits
from evaluate import prior_sample_stats
from model import bce_sum
from plots import tile


class AE(nn.Module):
    latent_dim = 2

    def __init__(self):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 2))
        self.dec = nn.Sequential(nn.Linear(2, 128), nn.ReLU(), nn.Linear(128, 64))

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        return self.dec(self.enc(x))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/q4"))
    ap.add_argument("--epochs", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = load_splits()
    xtr, xte = torch.from_numpy(s.x_train), torch.from_numpy(s.x_test)
    torch.manual_seed(42)
    ae = AE()
    opt = torch.optim.Adam(ae.parameters(), 1e-3)
    for ep in range(1, a.epochs + 1):
        perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(42 + ep))
        for i in range(0, len(xtr), 128):
            xb = xtr[perm[i:i + 128]]
            opt.zero_grad()
            bce_sum(ae(xb), xb).mean().backward()
            opt.step()
    ae.eval()
    clf = LogisticRegression(max_iter=2000).fit(s.x_train, s.y_train)
    with torch.no_grad():
        codes = ae.enc(xte)
        recon = float(bce_sum(ae(xte), xte).mean())
    res = {"recon_nll": recon, "code_mean": codes.mean(0).tolist(), "code_std": codes.std(0).tolist(),
           **prior_sample_stats(ae, clf)}
    (a.out / "q4.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    zp = torch.randn(N_PRIOR, 2, generator=torch.Generator().manual_seed(EVAL_SEED))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
    h = ax[0].scatter(codes[:, 0], codes[:, 1], c=s.y_test, cmap="tab10", s=8)
    ax[0].add_patch(plt.Circle((0, 0), 1, fill=False, ls="--", color="k"))
    ax[0].set_title("AE codes of the test set")
    ax[1].scatter(codes[:, 0], codes[:, 1], s=6, c="tab:blue", alpha=0.5, label="AE codes (test)")
    ax[1].scatter(zp[:300, 0], zp[:300, 1], s=6, c="tab:red", label="z ~ N(0,I) (300)")
    ax[1].legend()
    ax[1].set_title("prior samples vs data codes")
    for a_ in ax[:2]:
        a_.set_aspect("equal")
        a_.grid(alpha=0.25)
    with torch.no_grad():
        probs = torch.sigmoid(ae.decode(torch.randn(32, 2, generator=torch.Generator().manual_seed(7)))).numpy()
    ax[2].imshow(tile(probs, 4, 8), cmap="gray", vmin=0, vmax=1)
    ax[2].axis("off")
    ax[2].set_title("AE decoder on z ~ N(0,I)")
    fig.savefig(a.out / "q4_ae_prior.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

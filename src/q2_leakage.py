"""Q2 demo: target leakage vs a correctly masked autoregressive model on the binary digits.

    python src/q2_leakage.py --out results/q2

Both models are 64 -> 128 -> 64 MLPs trained with BCE. The leaky one sees the whole x and
predicts the same pixel; the masked one (MADE style) lets output i depend on pixels < i only.
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
import torch.nn.functional as F

from data import load_splits
from evaluate import independent_bernoulli_nll


class Leaky(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, 64))

    def forward(self, x):
        return self.net(x)


class Masked(nn.Module):
    """Output i is connected to hidden units of degree < i, which see only inputs <= their degree."""

    def __init__(self):
        super().__init__()
        self.l1, self.l2 = nn.Linear(64, 128), nn.Linear(128, 64)
        din, dh = torch.arange(1, 65), (torch.arange(128) % 63) + 1
        self.register_buffer("m1", (dh[:, None] >= din[None, :]).float())
        self.register_buffer("m2", (din[:, None] > dh[None, :]).float())

    def forward(self, x):
        h = F.relu(F.linear(x, self.l1.weight * self.m1, self.l1.bias))
        return F.linear(h, self.l2.weight * self.m2, self.l2.bias)


def nll(model, x):
    return F.binary_cross_entropy_with_logits(model(x), x, reduction="none").sum(1).mean()


def build(cls, seed=42):
    """Seed first, then initialise, so the whole experiment is reproducible."""
    torch.manual_seed(seed)
    return cls()


def fit(model, x, epochs=500, seed=42):
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for ep in range(1, epochs + 1):
        perm = torch.randperm(len(x), generator=torch.Generator().manual_seed(seed + ep))
        for i in range(0, len(x), 128):
            opt.zero_grad()
            nll(model, x[perm[i:i + 128]]).backward()
            opt.step()
    return model.eval()


@torch.no_grad()
def prefix_change(model, x, n=200):
    """Largest change of logit i when only x[i:] is randomised. 0 means causal."""
    gen, worst = torch.Generator().manual_seed(0), 0.0
    x = x[:n]
    for i in (0, 5, 20, 40, 63):
        x2 = x.clone()
        x2[:, i:] = (torch.rand(n, 64 - i, generator=gen) < 0.5).float()
        worst = max(worst, float((model(x)[:, i] - model(x2)[:, i]).abs().max()))
    return worst


@torch.no_grad()
def generate(model, n=16, seed=1):
    """Sample pixel by pixel, feeding the sampled prefix back in."""
    gen, z = torch.Generator().manual_seed(seed), torch.zeros(n, 64)
    for i in range(64):
        z[:, i] = (torch.rand(n, generator=gen) < torch.sigmoid(model(z)[:, i])).float()
    return z


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=Path("results/q2"))
    ap.add_argument("--epochs", type=int, default=500)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s = load_splits()
    xtr, xte = torch.from_numpy(s.x_train), torch.from_numpy(s.x_test)
    leaky, masked = fit(build(Leaky), xtr, a.epochs), fit(build(Masked), xtr, a.epochs)
    res = {m: {"train_nll": float(nll(net, xtr)), "test_nll": float(nll(net, xte)),
               "max_logit_change_when_future_changes": prefix_change(net, xte)}
           for m, net in (("leaky", leaky), ("masked", masked))}
    res["independent_bernoulli_test_nll"] = independent_bernoulli_nll(s.x_train, s.x_test)
    (a.out / "q2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))

    fig, ax = plt.subplots(3, 16, figsize=(16, 3.6))
    rows = [("real test images", s.x_test[:16]), ("leaky model, generated pixel by pixel", generate(leaky).numpy()),
            ("masked autoregressive model, generated pixel by pixel", generate(masked).numpy())]
    for r, (title, imgs) in enumerate(rows):
        for c in range(16):
            ax[r, c].imshow(imgs[c].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
            ax[r, c].axis("off")
        ax[r, 0].set_title(title, loc="left", fontsize=9)
    fig.savefig(a.out / "q2_leak_vs_masked.png", dpi=120, bbox_inches="tight")


if __name__ == "__main__":
    main()

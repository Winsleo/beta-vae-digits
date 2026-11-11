"""All figures of the report, built from saved models.

    python src/plots.py --models runs/main/models --out results/figs \
        [--early runs/main/models_ep100] [--history runs/main/history]

Pixels are shown as sigmoid probabilities (not Bernoulli samples). Reconstructions decode the
encoder mean. The prior samples use the same 16 z for every beta.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.datasets import load_digits

from config import BETAS, EVAL_SEED, beta_label, beta_tag
from data import load_splits
from evaluate import load_model

LABELS = [f"beta={beta_label(b)}" for b in BETAS]


def tile(imgs, rows, cols, pad=1):
    """Arrange 8x8 images in a rows x cols grid with a white gap."""
    canvas = np.ones((rows * 9 + pad, cols * 9 + pad))
    for r in range(rows):
        for c in range(cols):
            canvas[pad + r * 9: pad + r * 9 + 8, pad + c * 9: pad + c * 9 + 8] = imgs[r * cols + c].reshape(8, 8)
    return canvas


def row_of_images(axes, imgs):
    for ax, im in zip(axes, imgs):
        ax.imshow(np.asarray(im).reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        ax.axis("off")


def load_all(models_dir):
    return {b: load_model(Path(models_dir) / f"{beta_tag(b)}.pt") for b in BETAS}


@torch.no_grad()
def decode_probs(model, z):
    return torch.sigmoid(model.decode(z)).numpy()


def fig_data(s, out):
    digits = load_digits()
    fig, ax = plt.subplots(4, 10, figsize=(10, 4.4))
    for d in range(10):
        idx = np.where(s.y_train == d)[0][:3]
        ax[0, d].imshow(digits.data[s.idx_train[idx[0]]].reshape(8, 8) / 16, cmap="gray", vmin=0, vmax=1)
        for r in range(3):
            ax[r + 1, d].imshow(s.x_train[idx[r]].reshape(8, 8), cmap="gray", vmin=0, vmax=1)
        for r in range(4):
            ax[r, d].axis("off")
    ax[0, 0].set_title("gray (/16)", loc="left", fontsize=8)
    ax[1, 0].set_title("binarised (>=0.5), 3 per class", loc="left", fontsize=8)
    fig.savefig(out / "data.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def fig_recon(models, s, out):
    x = torch.from_numpy(s.x_test)
    first = [int(np.where(s.y_test == d)[0][0]) for d in range(10)]
    fig, ax = plt.subplots(5, 10, figsize=(10, 5.6))
    row_of_images(ax[0], s.x_test[first])
    ax[0, 0].set_title("input", loc="left", fontsize=9)
    for r, (b, m) in enumerate(models.items(), 1):
        with torch.no_grad():
            mu, _ = m.encode(x[first])
        row_of_images(ax[r], decode_probs(m, mu))
        ax[r, 0].set_title(f"beta={beta_label(b)} recon (decode mu)", loc="left", fontsize=9)
    fig.savefig(out / "recon.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_prior(models, out):
    z = torch.randn(16, 2, generator=torch.Generator().manual_seed(EVAL_SEED))
    fig, ax = plt.subplots(4, 16, figsize=(16, 4.4))
    for r, (b, m) in enumerate(models.items()):
        row_of_images(ax[r], decode_probs(m, z))
        ax[r, 0].set_title(f"beta={beta_label(b)}", loc="left", fontsize=10)
    fig.suptitle("Decoded prior samples z ~ N(0, I) (same 16 z for every beta)")
    fig.savefig(out / "prior.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def scatter(ax, m, s, title):
    with torch.no_grad():
        mu, _ = m.encode(torch.from_numpy(s.x_test))
    h = ax.scatter(mu[:, 0], mu[:, 1], c=s.y_test, cmap="tab10", s=8)
    ax.add_patch(plt.Circle((0, 0), 1, fill=False, ls="--", color="k", lw=0.8))
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    return h


def fig_scatter(models, s, out):
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.4), sharex=True, sharey=True)
    for a, (b, m) in zip(ax, models.items()):
        h = scatter(a, m, s, f"beta={beta_label(b)}")
    ax[0].set_xlim(-12, 12)
    ax[0].set_ylim(-12, 12)
    fig.colorbar(h, ax=ax, ticks=range(10), label="digit (colour only)", fraction=0.015)
    fig.suptitle("Encoder means of the test set, same axes for all betas (dashed: prior 1-sigma circle)")
    fig.savefig(out / "scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_interp(models, s, out):
    a_i, b_i = (int(np.where(s.y_test == d)[0][0]) for d in (0, 1))   # first 0 and first 1 of the test set
    x = torch.from_numpy(s.x_test)
    alpha = torch.linspace(0, 1, 11)[:, None]
    fig, ax = plt.subplots(5, 11, figsize=(12, 6.2))
    for a in ax[0]:
        a.axis("off")
    ax[0, 0].imshow(s.x_test[a_i].reshape(8, 8), cmap="gray")
    ax[0, 10].imshow(s.x_test[b_i].reshape(8, 8), cmap="gray")
    ax[0, 0].set_title("x_a", fontsize=9)
    ax[0, 10].set_title("x_b", fontsize=9)
    for r, (b, m) in enumerate(models.items(), 1):
        with torch.no_grad():
            mu, _ = m.encode(x[[a_i, b_i]])
        row_of_images(ax[r], decode_probs(m, (1 - alpha) * mu[0] + alpha * mu[1]))
        ax[r, 0].set_title(f"beta={beta_label(b)}", loc="left", fontsize=9)
    fig.suptitle("Linear interpolation between encoder means (alpha = 0 ... 1)")
    fig.savefig(out / "interpolation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_manifold(models, out):
    lin = np.linspace(-3, 3, 9)
    z = torch.tensor([[u, v] for v in lin[::-1] for u in lin], dtype=torch.float32)
    fig, ax = plt.subplots(1, 4, figsize=(17, 4.6))
    for a, (b, m) in zip(ax, models.items()):
        a.imshow(tile(decode_probs(m, z), 9, 9), cmap="gray", vmin=0, vmax=1)
        a.axis("off")
        a.set_title(f"beta={beta_label(b)}: decode on a 9x9 grid in [-3,3]^2")
    fig.savefig(out / "manifold.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def fig_history(history_dir, out):
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for b in BETAS:
        h = json.loads((Path(history_dir) / f"{beta_tag(b)}.json").read_text())
        ep = [r["epoch"] for r in h]
        ax[0].plot(ep, [r["val_bce"] for r in h], label=f"beta={beta_label(b)}")
        ax[1].plot(ep, [r["val_kl"] for r in h], label=f"beta={beta_label(b)}")
    ax[0].set_title("validation BCE (1 posterior sample)")
    ax[1].set_title("validation KL (closed form)")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
    fig.suptitle("Monitoring only: the validation set is never used to select a model")
    fig.savefig(out / "val_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_early_vs_late(early, late, s, out, names=("early", "late")):
    """Scatter, prior samples and reconstructions of two training budgets side by side."""
    x = torch.from_numpy(s.x_test)
    fig, ax = plt.subplots(2, 4, figsize=(15, 7.4), sharex=True, sharey=True)
    for r, models in enumerate((early, late)):
        for c, (b, m) in enumerate(models.items()):
            scatter(ax[r, c], m, s, f"beta={beta_label(b)}, {names[r]}")
    ax[0, 0].set_xlim(-12, 12)
    ax[0, 0].set_ylim(-12, 12)
    fig.savefig(out / "early_vs_late_scatter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    z = torch.randn(16, 2, generator=torch.Generator().manual_seed(EVAL_SEED))
    first = [int(np.where(s.y_test == d)[0][0]) for d in range(10)]
    fig, ax = plt.subplots(8, 16, figsize=(16, 8.6))
    for k, b in enumerate(BETAS):
        for j, models in enumerate((early, late)):
            row_of_images(ax[2 * k + j], decode_probs(models[b], z))
            ax[2 * k + j, 0].set_title(f"beta={beta_label(b)}  {names[j]}", loc="left", fontsize=9)
    fig.savefig(out / "early_vs_late_prior.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(9, 10, figsize=(10, 9.4))
    row_of_images(ax[0], s.x_test[first])
    r = 1
    for b in BETAS:
        for j, models in enumerate((early, late)):
            with torch.no_grad():
                mu, _ = models[b].encode(x[first])
            row_of_images(ax[r], decode_probs(models[b], mu))
            ax[r, 0].set_title(f"beta={beta_label(b)} {names[j]}", loc="left", fontsize=9)
            r += 1
    fig.savefig(out / "early_vs_late_recon.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("results/figs"))
    ap.add_argument("--early", type=Path, default=None, help="models of a shorter run, for side-by-side figures")
    ap.add_argument("--history", type=Path, default=None, help="history dir written by train.py")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    s, models = load_splits(), load_all(a.models)
    fig_data(s, a.out)
    fig_recon(models, s, a.out)
    fig_prior(models, a.out)
    fig_scatter(models, s, a.out)
    fig_interp(models, s, a.out)
    fig_manifold(models, a.out)
    if a.history:
        fig_history(a.history, a.out)
    if a.early:
        fig_early_vs_late(load_all(a.early), models, s, a.out)
    print(f"figures written to {a.out}/")


if __name__ == "__main__":
    main()

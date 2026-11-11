"""Evaluate trained models on the test set.

    python src/evaluate.py --models runs/main/models --out results/main.json

Per beta it reports the reconstruction NLL (20 posterior samples), the closed-form KL
(total and per latent dimension), the standard negative ELBO (= recon NLL + KL), an
IWAE estimate of -log p(x), statistics of the posterior, and how diverse the decoded
prior samples are (class entropy under a logistic-regression digit classifier).
The weighted training loss (BCE + beta * KL) is never used to compare betas.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression

from config import BETAS, EVAL_SEED, MC_SAMPLES, N_PRIOR, beta_label, beta_tag
from data import load_splits
from model import VAE, bce_sum, kl_divergence


def load_model(path) -> VAE:
    model = VAE()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.eval()


@torch.no_grad()
def reconstruction_metrics(model: VAE, x: torch.Tensor) -> dict:
    """Reconstruction NLL (MC over the posterior), KL and the standard negative ELBO."""
    mu, logvar = model.encode(x)
    kl_dim = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar)
    gen = torch.Generator().manual_seed(EVAL_SEED)
    acc = torch.zeros(len(x))
    for _ in range(MC_SAMPLES):
        z = mu + torch.exp(0.5 * logvar) * torch.randn(mu.shape, generator=gen)
        acc += bce_sum(model.decode(z), x)
    recon, kl = float((acc / MC_SAMPLES).mean()), float(kl_dim.sum(1).mean())
    return {"recon_nll": recon, "kl": kl, "kl_per_dim": kl_dim.mean(0).tolist(), "neg_elbo": recon + kl}


@torch.no_grad()
def iwae_nll(model: VAE, x: torch.Tensor, k: int = 1000, seed: int = 0) -> float:
    """Importance-weighted estimate of -log p(x): tighter than the negative ELBO."""
    gen = torch.Generator().manual_seed(seed)
    mu, logvar = model.encode(x)
    std, d = torch.exp(0.5 * logvar), mu.shape[1]
    const = 0.5 * d * math.log(2 * math.pi)
    out = []
    for i in range(len(x)):
        eps = torch.randn(k, d, generator=gen)
        z = mu[i] + std[i] * eps
        log_px_z = -bce_sum(model.decode(z), x[i].expand(k, -1))
        log_pz = -0.5 * z.square().sum(1) - const
        log_q = -0.5 * eps.square().sum(1) - 0.5 * logvar[i].sum() - const
        out.append(torch.logsumexp(log_px_z + log_pz - log_q, 0) - math.log(k))
    return -float(torch.stack(out).mean())


@torch.no_grad()
def posterior_stats(model: VAE, x: torch.Tensor) -> dict:
    """Spread of the encoder means, mean posterior std, and the aggregate-posterior variance."""
    mu, logvar = model.encode(x)
    return {
        "mu_std": mu.std(0).tolist(),
        "post_std": torch.exp(0.5 * logvar).mean(0).tolist(),
        "aggregate_var": (mu.var(0) + logvar.exp().mean(0)).tolist(),  # prior variance is 1
    }


@torch.no_grad()
def prior_sample_stats(model: VAE, clf: LogisticRegression, n: int = N_PRIOR) -> dict:
    """Decode z ~ N(0,I), binarise, classify; low class entropy = samples collapse onto few digits."""
    z = torch.randn(n, model.latent_dim, generator=torch.Generator().manual_seed(EVAL_SEED))
    pred = clf.predict((torch.sigmoid(model.decode(z)) >= 0.5).numpy().astype(np.float32))
    hist = np.bincount(pred, minlength=10) / n
    nz = hist[hist > 0]
    return {"class_hist": hist.round(4).tolist(), "class_entropy": float(-(nz * np.log(nz)).sum())}


def independent_bernoulli_nll(x_train: np.ndarray, x_test: np.ndarray) -> float:
    """Reference model without latents: per-pixel Bernoulli fitted on the training set."""
    p = np.clip(x_train.mean(0), 1e-4, 1 - 1e-4)
    return float((-(x_test * np.log(p) + (1 - x_test) * np.log(1 - p)).sum(1)).mean())


def evaluate_dir(models_dir: Path, iwae_k: int = 1000) -> dict:
    s = load_splits()
    clf = LogisticRegression(max_iter=2000).fit(s.x_train, s.y_train)
    x = torch.from_numpy(s.x_test)
    res = {"classifier_test_acc": float(clf.score(s.x_test, s.y_test)),
           "independent_bernoulli_nll": independent_bernoulli_nll(s.x_train, s.x_test), "betas": {}}
    for beta in BETAS:
        model = load_model(Path(models_dir) / f"{beta_tag(beta)}.pt")
        r = reconstruction_metrics(model, x)
        if iwae_k:
            r["iwae_nll"] = iwae_nll(model, x, iwae_k)
        r.update(posterior_stats(model, x))
        r.update(prior_sample_stats(model, clf))
        res["betas"][beta_label(beta)] = r
    return res


def to_markdown(res: dict) -> str:
    rows = ["| beta | recon NLL | KL | -ELBO | IWAE | prior class entropy |", "|---|---:|---:|---:|---:|---:|"]
    for b, r in res["betas"].items():
        rows.append(f"| {b} | {r['recon_nll']:.4f} | {r['kl']:.4f} | {r['neg_elbo']:.4f} | "
                    f"{r.get('iwae_nll', float('nan')):.2f} | {r['class_entropy']:.2f} |")
    rows.append(f"\nindependent-Bernoulli baseline NLL: {res['independent_bernoulli_nll']:.2f}")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", type=Path, required=True, help="directory with beta*.pt")
    ap.add_argument("--out", type=Path, default=None, help="write the full results as JSON")
    ap.add_argument("--iwae-k", type=int, default=1000, help="IWAE samples per image, 0 to skip")
    a = ap.parse_args()
    res = evaluate_dir(a.models, a.iwae_k)
    print(to_markdown(res))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(res, indent=1))
        print(f"\nwritten to {a.out}")


if __name__ == "__main__":
    main()

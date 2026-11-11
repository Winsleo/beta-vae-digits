"""Acceptance checks. Run with `python tests/test_core.py` or `pytest tests`."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data import load_binary_digits, load_splits, split_indices  # noqa: E402
from evaluate import iwae_nll, reconstruction_metrics  # noqa: E402
from model import VAE, beta_vae_loss, bce_sum, kl_divergence  # noqa: E402


def test_kl_is_zero_for_standard_normal():
    assert torch.allclose(kl_divergence(torch.zeros(5, 2), torch.zeros(5, 2)), torch.zeros(5), atol=1e-6)


def test_kl_is_non_negative():
    g = torch.Generator().manual_seed(0)
    mu, logvar = torch.randn(1000, 2, generator=g), 2 * torch.randn(1000, 2, generator=g)
    assert (kl_divergence(mu, logvar) >= -1e-6).all()


def test_kl_matches_closed_form_example():
    # mu=1, sigma^2=1 -> 0.5 ; mu=0, logvar=1 -> 0.5*(e-2)
    kl = kl_divergence(torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]]))
    assert abs(float(kl) - (0.5 + 0.5 * (np.e - 2))) < 1e-6


def test_encoder_gets_gradient_through_reparameterisation():
    """Reconstruction loss only (no KL), so the gradient must come through the sampled z."""
    torch.manual_seed(0)
    model, x = VAE(), torch.rand(5, 64)
    logits, _, _, z = model(x)
    assert z.requires_grad
    bce_sum(logits, x).mean().backward()
    encoder_params = list(model.encoder.parameters()) + list(model.mu_head.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder_params)


def test_loss_is_batch_mean_of_bce_plus_beta_kl():
    torch.manual_seed(0)
    model, x = VAE(), (torch.rand(8, 64) > 0.5).float()
    logits, mu, logvar, _ = model(x)
    loss, bce, kl = beta_vae_loss(logits, x, mu, logvar, 4.0)
    assert bce.shape == kl.shape == (8,)
    assert torch.allclose(loss, (bce + 4.0 * kl).mean())


def test_binarisation_threshold_is_inclusive():
    x, _ = load_binary_digits()
    assert x.shape == (1797, 64) and set(np.unique(x)) <= {0.0, 1.0}
    assert (8 / 16.0 >= 0.5) and not (7 / 16.0 >= 0.5)   # pixel value 8 -> 1, 7 -> 0


def test_split_is_stratified_disjoint_and_deterministic():
    _, y = load_binary_digits()
    tr, va, te = split_indices(y)
    assert (len(tr), len(va), len(te)) == (1078, 359, 360)
    assert len(set(tr) | set(va) | set(te)) == len(y)
    assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
    for idx, share in ((tr, 0.6), (va, 0.2), (te, 0.2)):          # every class keeps ~its share
        for c in range(10):
            assert abs(np.mean(y[idx] == c) * len(idx) / np.sum(y == c) - share) < 0.01
    tr2, va2, te2 = split_indices(y)
    assert all(np.array_equal(a, b) for a, b in ((tr, tr2), (va, va2), (te, te2)))


def test_iwae_is_not_looser_than_elbo_on_a_random_model():
    torch.manual_seed(0)
    model = VAE().eval()
    x = torch.from_numpy(load_splits().x_test[:20])
    assert iwae_nll(model, x, k=200) <= reconstruction_metrics(model, x)["neg_elbo"] + 0.5


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\nall {len(tests)} checks passed")

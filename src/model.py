"""beta-VAE for 8x8 binary digits, plus the loss terms used everywhere else."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import HIDDEN_DIM, INPUT_DIM, LATENT_DIM


class VAE(nn.Module):
    """Encoder 64 -> 128 -> (mu, logvar); decoder 2 -> 128 -> 64 logits.

    The layers are created in this exact order: it fixes the random initialisation,
    and therefore the reported numbers, for a given seed.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(nn.Linear(INPUT_DIM, hidden_dim), nn.ReLU())
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, INPUT_DIM)
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.mu_head(h), self.logvar_head(h)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps, eps ~ N(0, I)."""
        return mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar, z


def kl_divergence(mu, logvar):
    """KL(N(mu, diag(exp(logvar))) || N(0, I)), summed over latent dims -> shape (B,)."""
    return 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).sum(dim=1)


def bce_sum(logits, x):
    """Bernoulli negative log-likelihood, summed over the 64 pixels -> shape (B,)."""
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").sum(dim=1)


def beta_vae_loss(logits, x, mu, logvar, beta: float):
    """Batch mean of (BCE_sum + beta * KL). Returns (loss, bce_per_image, kl_per_image)."""
    bce, kl = bce_sum(logits, x), kl_divergence(mu, logvar)
    return (bce + beta * kl).mean(), bce, kl

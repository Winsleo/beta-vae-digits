import torch
import torch.nn as nn

class VAE(nn.Module):
    """
    β-VAE with 2D latent space.
    Encoder: 64 -> 128 (ReLU) -> μ(2) + logσ²(2)
    Decoder: 2 -> 128 (ReLU) -> 64 logits (for BCE with logits)
    """
    def __init__(self, latent_dim=2):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(128, latent_dim)
        self.logvar_head = nn.Linear(128, latent_dim)

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),  # Output logits, no sigmoid
        )

    def encode(self, x):
        """Encode to latent parameters."""
        h = self.encoder(x)
        mu = self.mu_head(h)
        logvar = self.logvar_head(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """Reparameterization trick: z = μ + σ * ε."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z

    def decode(self, z):
        """Decode latent z to logits."""
        return self.decoder(z)

    def forward(self, x):
        """Forward pass: return reconstruction logits and latent params."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar, z

    def compute_kl(self, mu, logvar):
        """Compute KL divergence: D_KL(q(z|x) || p(z))."""
        # KL = 0.5 * sum_j [μ_j^2 + exp(logvar_j) - 1 - logvar_j]
        kl = 0.5 * torch.sum(mu**2 + torch.exp(logvar) - 1 - logvar, dim=1)
        return kl

    def compute_bce_sum(self, logits, x):
        """Compute BCE summed over 64 pixels (not averaged)."""
        # BCE = -sum_i [x_i*log(p_i) + (1-x_i)*log(1-p_i)]
        # Using BCEWithLogitsLoss: loss = -[x*log(sigmoid(logits)) + (1-x)*log(1-sigmoid(logits))]
        # BCEWithLogitsLoss with reduction='none' gives per-sample loss; we need per-dimension
        bce_per_sample = nn.BCEWithLogitsLoss(reduction='none')(logits, x)  # shape: (B, 64)
        bce_sum = torch.sum(bce_per_sample, dim=1)  # shape: (B,)
        return bce_sum

def kl_correctness_check():
    """Verify KL calculation: μ=0, logσ²=0 => KL=0."""
    model = VAE()
    mu = torch.zeros(5, 2)
    logvar = torch.zeros(5, 2)
    kl = model.compute_kl(mu, logvar)
    expected = torch.zeros(5)
    assert torch.allclose(kl, expected, atol=1e-6), f"KL sanity check failed: {kl}"
    print("✓ KL correctness check 1: μ=0, logσ²=0 => KL=0")

def kl_nonneg_check():
    """Verify KL≥0 for valid Gaussian parameters."""
    model = VAE()
    mu = torch.randn(10, 2)
    logvar = torch.randn(10, 2) * 2  # Keep logvar bounded
    kl = model.compute_kl(mu, logvar)
    assert torch.all(kl >= -1e-6), f"KL negativity check failed: min KL = {kl.min()}"
    print("✓ KL correctness check 2: KL≥0 for valid Gaussian parameters")

def gradient_backprop_check():
    """Verify reconstruction loss sends gradients through reparameterization to the encoder.

    Backward is reconstruction-only. KL also depends on the encoder, so including it
    would not show that the reparameterized sample path itself carries gradients.
    """
    model = VAE()
    x = torch.rand(5, 64)
    logits, mu, logvar, z = model(x)
    assert z.requires_grad, "reparameterized z does not require grad"

    bce = torch.mean(model.compute_bce_sum(logits, x))
    bce.backward()

    encoder_params = list(model.encoder.parameters()) + list(model.mu_head.parameters())
    encoder_has_grad = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in encoder_params
    )
    assert encoder_has_grad, "Gradients not flowing to encoder through reparameterization"
    print("✓ Gradient backprop check: reconstruction loss backpropagates to encoder")

if __name__ == "__main__":
    kl_correctness_check()
    kl_nonneg_check()
    gradient_backprop_check()
    print("\n✓ All correctness checks passed!")

"""Shared constants. Every value here is part of the experimental protocol."""

BETAS = (0.0, 0.1, 1.0, 4.0)

# model
LATENT_DIM = 2
HIDDEN_DIM = 128
INPUT_DIM = 64

# optimisation
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 500
TRAIN_SEED = 42          # model init and per-epoch shuffling

# data
SPLIT_SEED = 42          # stratified 60/20/20 split

# evaluation
EVAL_SEED = 12345        # posterior MC noise, prior samples
MC_SAMPLES = 20          # posterior samples for the reconstruction NLL
N_PRIOR = 2000           # prior samples for the class-entropy check

_TAGS = {0.0: "beta0", 0.1: "beta0p1", 1.0: "beta1", 4.0: "beta4"}


def beta_tag(beta: float) -> str:
    """File-name friendly tag, e.g. 0.1 -> 'beta0p1'."""
    return _TAGS[float(beta)]


def beta_label(beta: float) -> str:
    return f"{beta:g}"

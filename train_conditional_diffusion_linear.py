"""
train_conditional_diffusion_linear.py

Duplicate of train_conditional_diffusion.py with the storage domain and the loss
domain separated: the diffusion model operates on LINEAR flux patches, while the
loss is measured through the asinh stretch.

Why
---
Training on asinh-stretched patches puts the DDPM's state space in a compressed
coordinate. Noise is added and removed in asinh units, and on the way back to
flux an error dz is amplified by dflux/dz = beta*cosh(...), which reaches ~14.5
mag^-1 at the bright end of this dataset. Bright-source artifacts are therefore
manufactured by the coordinate change itself, not just by the model.

Here the patches are linear (see gen_data_linear.py), so the forward process adds
noise in units of flux and the model behaves like a standard DDPM. The asinh map
survives only as the loss weighting, which is what it was actually good for: it
is roughly uniform in magnitude error, so faint sources still generate gradient
alongside bright ones.

The two losses
--------------
    eps term      MSE(eps_pred, eps)                     -- the ordinary DDPM loss
    stretch term  MSE(g(ideal_hat), g(ideal))            -- asinh-domain fidelity

where g is `ideal_loss_stretch` from the dataset's norm.json and
    ideal_hat = (x_t - sqrt(1-abar)*eps_pred) / sqrt(abar)
is the model's implied clean estimate.

ideal_hat is deliberately NOT clamped. At high t, sqrt(abar) -> 0 amplifies the
eps error by sqrt((1-abar)/abar) -- 6.4x at t=900, 2e4x at t=999 -- which looks
like it needs bounding, but asinh bounds it already: measured on this dataset,
max|g(ideal_hat)| is 0.93 at t=0 and still only 1.85 at t=999, and the gradient
w.r.t. eps_pred grows just 8x (4.1e-6 -> 3.4e-5) across the whole schedule. The
log growth of asinh self-attenuates the high-t garbage.

An earlier version clamped ideal_hat to the data range [0, 1]. That was actively
harmful and is worth recording: true pixel values sit at ~5e-6 in stored units,
so ideal_hat is distributed symmetrically about ~0 and the lower bound truncated
~50% of pixels at EVERY timestep, t=0 included. The truncation is one-sided --
an overestimate gets a gradient pushing it down, an underestimate below zero gets
none -- so the term acquired a systematic downward push, i.e. exactly the flux
shrinkage this denoiser is already prone to. Do not reintroduce a lower clamp.

The eps term is kept as the backbone because the stretch term saturates at the
faint end: the asinh weight flattens for pixels fainter than the diffusion noise,
and on m31bK50 with a 100th-percentile (min-max) anchor the t=0 noise is 0.0064
in stored units = raw flux 11.4, against a 99.9th percentile of 6.09. With that
anchor the stretch term is driven almost entirely by the brightest ~0.05% of
pixels and the eps term does all the faint-end work. Lowering `anchor_percentile`
in gen_data_linear.py is what changes that balance.

The two weights are separate (not a convex mix) because the terms have different
natural scales; both are printed every epoch so the ratio can be set from a first
run rather than guessed. At init they happen to land within 3x of each other
(eps 1.00, stretch 0.36), which is why both default to 1.0.

Null token
----------
The conditioning dropout token is NOT zeros here. In the linear domain a real
observed patch sits at ~2e-4 in stored units (median raw 552 against a min-max
span of ~99500), so an all-zeros patch is indistinguishable from a genuine faint
one and classifier-free guidance would train against a token the model cannot
recognize. NULL_TOKEN is -1.0, outside the data range by construction. It is
recorded in the checkpoint; downstream consumers that call model(x_t, zeros, t)
for the unconditional path MUST switch to this value.

Usage
-----
    cd ~/noir_ml/mycode
    python train_conditional_diffusion_linear.py \
        --data-dir <gen_data_linear output dir> \
        --epochs 100 --batch-size 32

    # Sanity check first (memorize a single batch; loss should -> ~0):
    python train_conditional_diffusion_linear.py --overfit-one-batch

Requires: torch, numpy. Optional: ema_pytorch (falls back to a built-in
EMA if absent).
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# Conditioning value standing in for "no conditioning" under classifier-free
# guidance dropout. Must lie outside the stored data range; see module docstring.
NULL_TOKEN = -1.0


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class NpyPairDataset(Dataset):
    """Yields (ideal, observed) tensors straight from the generated arrays.

    ideal    -> the sharp target the diffusion model learns to generate
    observed -> the blurry conditioning channel

    Values pass through untouched -- the arrays on disk are already in their
    final (generator-normalized) linear domain and this class must never
    rescale, stretch, or otherwise re-map them. Augmentation (if enabled) is a
    random D4 element applied identically to both members of the pair, so the
    observed/ideal registration is preserved.
    """

    def __init__(self, observed: np.ndarray, ideal: np.ndarray,
                 augment: bool = False):
        if observed.shape != ideal.shape:
            sys.exit(f"[data] observed {observed.shape} and ideal "
                     f"{ideal.shape} arrays must have identical shapes")
        self.observed = np.ascontiguousarray(observed, dtype=np.float32)
        self.ideal = np.ascontiguousarray(ideal, dtype=np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.observed)

    def __getitem__(self, idx):
        observed = self.observed[idx]   # (1, H, W)
        ideal = self.ideal[idx]
        if self.augment:
            # torch's RNG is seeded per DataLoader worker, unlike numpy's.
            k = int(torch.randint(0, 8, (1,)).item())
            if k % 4:
                observed = np.rot90(observed, k % 4, axes=(-2, -1))
                ideal = np.rot90(ideal, k % 4, axes=(-2, -1))
            if k >= 4:
                observed = np.flip(observed, axis=-1)
                ideal = np.flip(ideal, axis=-1)
        return (
            torch.from_numpy(np.ascontiguousarray(ideal)),
            torch.from_numpy(np.ascontiguousarray(observed)),
        )


def load_split(data_dir: Path, split: str):
    observed = np.load(data_dir / f"{split}_observed.npy")
    ideal = np.load(data_dir / f"{split}_ideal.npy")
    print(f"[data] {split}: {len(observed)} pairs {tuple(observed.shape[1:])} "
          f"observed[{observed.min():+.4g}, {observed.max():+.4g}] "
          f"ideal[{ideal.min():+.4g}, {ideal.max():+.4g}]")
    return observed, ideal


# ----------------------------------------------------------------------------
# asinh loss stretch
# ----------------------------------------------------------------------------

def make_stretch(params):
    """Build g(u) from a norm.json `ideal_loss_stretch` entry.

        g(u) = (arcsinh((u - median) / beta) - lo_s) / (hi_s - lo_s)

    The parameters are the asinh map the original pipeline used as its storage
    domain, rebased onto linear-normalized values by gen_data_linear.py, so
    g(stored_linear_patch) equals the value the asinh dataset held to float32
    precision. Loss numbers are therefore comparable between the two pipelines.

    The lo_s offset cancels inside an MSE and the span only rescales it; both are
    kept so the numbers stay on the old scale rather than an arbitrary one.
    """
    if params.get("method") != "asinh":
        sys.exit(f"[loss] ideal_loss_stretch must be an asinh map, got "
                 f"{params.get('method')!r}")
    median = float(params["median"])
    beta = float(params["beta"])
    span = float(params["hi_s"]) - float(params["lo_s"])
    lo_s = float(params["lo_s"])

    def stretch(u: torch.Tensor) -> torch.Tensor:
        return (torch.asinh((u - median) / beta) - lo_s) / span

    return stretch


# ----------------------------------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal 2021, as used in the DPS paper)
# ----------------------------------------------------------------------------

def cosine_alpha_bar(T: int, s: float = 0.008,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """alpha_bar[t], t = 0..T-1. Always COMPUTED in float64.

    `dtype` sets only the return dtype. It defaults to float32 so training is
    bit-for-bit unchanged; the solvers pass float64 so that the schedule they
    derive sigma(t) from is not pre-rounded to ~1e-7 relative.
    """
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = betas.clamp(1e-8, 0.999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).to(dtype)


# ----------------------------------------------------------------------------
# Model: flat CNN denoiser with FiLM timestep conditioning
# ----------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        # This module has no parameters, so `model.to(float64)` had no way to
        # reach it and a hardcoded .float() would produce a float32 embedding
        # that then hit float64 Linear weights. A non-persistent buffer tracks
        # the module dtype without entering state_dict.
        self.register_buffer("_dtype_probe", torch.zeros(1), persistent=False)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        dt = self._dtype_probe.dtype
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=dt) / half
        )
        args = t.to(dt)[:, None] * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class FiLMConvBlock(nn.Module):
    """Conv -> GroupNorm -> FiLM(gamma, beta from t-embedding) -> SiLU."""

    def __init__(self, channels: int, dilation: int, t_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3,
                              padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(8, channels)
        self.film = nn.Linear(t_dim, 2 * channels)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)  # start as identity modulation

    def forward(self, x, t_emb):
        h = self.norm(self.conv(x))
        gamma, beta = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]
        return F.silu(h) + x  # residual


class ConditionalFlatCNN(nn.Module):
    """DnCNN-style flat denoiser, dilation pyramid 1,2,3,4,4,3,2,1
    (41-pixel receptive field), no downsampling. Predicts the noise eps.

    Input channels: [x_t (noisy ideal), y (observed conditioning)].
    The observed channel is NEVER noised -- it is a fixed conditioning signal at
    every timestep, which is what injects the pairing statistics. In the linear
    domain it carries more of the load than it did in asinh: faint sources sit
    below the diffusion noise floor at every t, so the conditioning channel is
    the only place their information survives at full precision.
    """

    DILATIONS = (1, 2, 3, 4, 4, 3, 2, 1)

    def __init__(self, channels: int = 64, t_dim: int = 128):
        super().__init__()
        self.t_embed = nn.Sequential(
            SinusoidalTimeEmbedding(t_dim),
            nn.Linear(t_dim, t_dim), nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )
        self.head = nn.Conv2d(2, channels, 3, padding=1)  # 2 in-channels
        self.blocks = nn.ModuleList(
            FiLMConvBlock(channels, d, t_dim) for d in self.DILATIONS
        )
        self.tail = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)  # predict ~0 noise at init

    def forward(self, x_t, y_cond, t):
        t_emb = self.t_embed(t)
        h = self.head(torch.cat([x_t, y_cond], dim=1))
        for block in self.blocks:
            h = block(h, t_emb)
        return self.tail(h)


# ----------------------------------------------------------------------------
# EMA (uses ema_pytorch if installed, else a minimal fallback)
# ----------------------------------------------------------------------------

class SimpleEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(),
                                                     alpha=1 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_dict(self):
        return self.shadow


def make_ema(model):
    try:
        from ema_pytorch import EMA
        return EMA(model, beta=0.999, update_every=1), True
    except ImportError:
        return SimpleEMA(model), False


# ----------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------

def diffusion_loss(model, ideal, observed, alpha_bar, device, stretch,
                   eps_weight=1.0, stretch_weight=1.0,
                   p_uncond=0.0, identity_frac=0.0, identity_t_max=100,
                   low_t_frac=0.0, low_t_max=100):
    """eps-prediction DDPM loss plus an asinh-domain fidelity term.

    Returns (total, eps_mse, stretch_mse) so the two components can be tracked
    separately -- their natural scales differ and the weights are set from the
    printed breakdown, not from theory.

    p_uncond > 0 enables CLASSIFIER-FREE GUIDANCE training: that fraction of
    the batch has its conditioning channel replaced by NULL_TOKEN, so the same
    weights learn BOTH
        eps_theta(x_t, y, t)     the conditional/posterior score, and
        eps_theta(x_t, null, t)  a genuine UNCONDITIONAL prior score.

    Why this matters for the PnP use: RED / Graikos regularizers assume the
    denoiser models the PRIOR p(x). A purely conditional denoiser gives the
    POSTERIOR score, which already contains the likelihood -- using it next to
    an explicit ||y - Ax||^2 term double-counts the measurement.
    """
    bsz = ideal.shape[0]
    T = alpha_bar.shape[0]
    t = torch.randint(0, T, (bsz,), device=device)
    if low_t_frac > 0:
        # LOW-t OVERSAMPLING: the RED/PnP prox only ever calls the denoiser at
        # t <= ~75, but uniform t gives that regime just low_t_max/T (~10%) of
        # the gradient signal. Redirect a fraction of the batch there. (Unlike
        # the identity rows below, these keep their noise -- they are ordinary
        # denoising problems, just concentrated where the model is used.)
        low = (torch.rand(bsz, device=device) < low_t_frac)
        t_low = torch.randint(0, max(int(low_t_max), 1), (bsz,),
                              device=device)
        t = torch.where(low, t_low, t)
    ab = alpha_bar[t][:, None, None, None]
    eps = torch.randn_like(ideal)
    if identity_frac > 0:
        # IDENTITY / fixed-point term: this fraction of the batch gets ZERO
        # noise, so x_t = sqrt(abar)*ideal and the target eps is exactly 0.
        # It teaches D(x, y) ~ x on a clean field.
        #
        # Why it matters here: the RED gradient is x - D(x), so if the model is
        # not a fixed point on the ideal manifold, RED pulls AWAY from the truth
        # even when x is already correct.
        #
        # The identity rows also get their t RESAMPLED into [0, identity_t_max).
        # A zero-noise draw at t=900 is just 0.03*ideal -- a near-black frame
        # that teaches nothing and never occurs in the prox, which runs at
        # t<=~75. Only the non-identity rows keep the uniform t.
        idm = (torch.rand(bsz, device=device) < identity_frac)
        eps = torch.where(idm[:, None, None, None],
                          torch.zeros_like(eps), eps)
        t_low = torch.randint(0, max(int(identity_t_max), 1), (bsz,),
                              device=device)
        t = torch.where(idm, t_low, t)
        ab = alpha_bar[t][:, None, None, None]
    x_t = ab.sqrt() * ideal + (1 - ab).sqrt() * eps
    if p_uncond > 0:
        drop = (torch.rand(bsz, device=device) < p_uncond)
        observed = torch.where(drop[:, None, None, None],
                               torch.full_like(observed, NULL_TOKEN), observed)
    eps_pred = model(x_t, observed, t)

    eps_mse = F.mse_loss(eps_pred, eps)
    if stretch_weight > 0:
        # The implied clean estimate. Its error is the eps error amplified by
        # sqrt((1-abar)/abar) -- 0.13 at t=75, 2e4 at t=999 -- but it is fed
        # straight into asinh, whose log growth keeps g(ideal_hat) bounded
        # (<1.9 even at t=999). Clamping it here would truncate ~50% of pixels
        # one-sidedly at every t; see the module docstring.
        ideal_hat = (x_t - (1 - ab).sqrt() * eps_pred) / ab.sqrt()
        stretch_mse = F.mse_loss(stretch(ideal_hat), stretch(ideal))
    else:
        stretch_mse = torch.zeros((), device=device)

    total = eps_weight * eps_mse + stretch_weight * stretch_mse
    return total, eps_mse, stretch_mse


@torch.no_grad()
def validate(model, loader, alpha_bar, device, stretch,
             eps_weight, stretch_weight):
    model.eval()
    totals = np.zeros(3)
    n = 0
    for ideal, observed in loader:
        ideal, observed = ideal.to(device), observed.to(device)
        parts = diffusion_loss(model, ideal, observed, alpha_bar, device,
                               stretch, eps_weight=eps_weight,
                               stretch_weight=stretch_weight)
        totals += np.array([p.item() for p in parts]) * ideal.shape[0]
        n += ideal.shape[0]
    model.train()
    return totals / max(n, 1)


def save_checkpoint(path, model, ema, ema_is_lib, optimizer, epoch,
                    dataset_norm, args):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": (ema.ema_model.state_dict() if ema_is_lib
                      else ema.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        # NO trainer-side transform exists ("transform" key is deliberately
        # absent so stale consumers fail loudly). The model operates in the
        # dataset's stored LINEAR domain; `dataset_norm` is the generator's
        # norm.json, which is the ONLY map between that domain and physical
        # flux. Inverting it is the consumer's job.
        "dataset_norm": dataset_norm,
        # The model's state space is linear; the asinh map appears only as the
        # training loss weighting. A consumer must NOT apply it to activations.
        "state_domain": "linear",
        "loss_domain": "asinh",
        "loss_stretch": (dataset_norm or {}).get("ideal_loss_stretch"),
        # Unconditional path is model(x_t, full_like(x_t, NULL_TOKEN), t).
        # Zeros are NOT the null token in the linear domain -- see module
        # docstring. Consumers must read this key rather than assume.
        "null_token": NULL_TOKEN,
        "data_dir": str(args.data_dir),
        "arch": {"channels": args.channels,
                 "dilations": list(ConditionalFlatCNN.DILATIONS),
                 "t_dim": 128, "in_channels": 2},
        "diffusion": {"timesteps": args.timesteps, "schedule": "cosine"},
        "eps_weight": args.eps_weight,
        "stretch_weight": args.stretch_weight,
        "p_uncond": args.p_uncond,
        "identity_frac": args.identity_frac,
        "identity_t_max": args.identity_t_max,
        "low_t_frac": args.low_t_frac,
        "low_t_max": args.low_t_max,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
    }
    torch.save(ckpt, path)


def main():
    ap = argparse.ArgumentParser(
        description="Train a conditional flat-CNN DDPM on LINEAR observed/ideal "
                    ".npy patch pairs with an asinh-domain loss "
                    "(gen_data_linear.py output).")
    ap.add_argument("--data-dir", type=Path, required=True,
                    help="directory holding {train,val}_{observed,ideal}.npy "
                         "and norm.json with an `ideal_loss_stretch` entry")
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("checkpoints_cond_diffusion_linear"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--eps-weight", type=float, default=1.0,
                    help="weight on the ordinary eps-prediction MSE. Keep it "
                         "non-zero: the asinh weight flattens for pixels "
                         "fainter than the diffusion noise, so this is the only "
                         "term carrying the faint end.")
    ap.add_argument("--stretch-weight", type=float, default=1.0,
                    help="weight on MSE in the asinh domain of the implied "
                         "clean estimate. Both components are printed every "
                         "epoch -- set this ratio from a first run rather than "
                         "guessing. 0 disables the term (and the extra "
                         "reconstruction), giving a plain linear-domain DDPM.")
    ap.add_argument("--p-uncond", type=float, default=0.15,
                    help="classifier-free-guidance dropout: fraction of "
                         "training samples whose observed conditioning is "
                         "replaced by NULL_TOKEN. >0 makes the SAME weights "
                         "usable as an unconditional prior -- required if you "
                         "want to use this model in a RED/Graikos regularizer "
                         "alongside an explicit ||y-Ax||^2 term without "
                         "double-counting the likelihood.")
    ap.add_argument("--identity-frac", type=float, default=0.15,
                    help="fraction of training draws with ZERO noise and target "
                         "eps=0, teaching D(x,y) ~ x on clean fields. Required "
                         "for the RED gradient x - D(x) to vanish on the truth.")
    ap.add_argument("--identity-t-max", type=int, default=100,
                    help="identity draws are placed at t ~ U[0, this). The "
                         "PnP/RED prox runs at t<=~75.")
    ap.add_argument("--low-t-frac", type=float, default=0.25,
                    help="fraction of NOISY training draws whose t is resampled "
                         "into [0, --low-t-max).")
    ap.add_argument("--low-t-max", type=int, default=100,
                    help="upper bound (exclusive) for the oversampled low-t "
                         "range; match to the largest t the prox uses.")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable the on-the-fly D4 (rot90/flip) pair "
                         "augmentation of the training split.")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--overfit-one-batch", action="store_true",
                    help="Sanity check: memorize a single batch and track a "
                         "deterministic fixed-noise eval loss.")
    ap.add_argument("--sanity-steps", type=int, default=3000,
                    help="Steps for the overfit-one-batch check.")
    ap.add_argument("--sanity-lr", type=float, default=3e-4,
                    help="Learning rate for the overfit-one-batch check.")
    ap.add_argument("--device", type=str,
                    default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device(args.device)
    print(f"[setup] device = {device}"
          + ("  (CPU: fine for --overfit-one-batch verification; use the GPU "
             "cluster for the full run)" if device.type == "cpu" else ""))

    # --- data -----------------------------------------------------------
    # Arrays are used exactly as stored: the generator owns the domain.
    train_ds = NpyPairDataset(*load_split(args.data_dir, "train"),
                              augment=not args.no_augment)
    val_ds = NpyPairDataset(*load_split(args.data_dir, "val"), augment=False)

    norm_path = args.data_dir / "norm.json"
    if not norm_path.exists():
        sys.exit(f"[data] no norm.json in {args.data_dir}. This trainer needs "
                 f"its `ideal_loss_stretch` entry to define the loss domain; "
                 f"regenerate the dataset with gen_data_linear.py.")
    dataset_norm = json.loads(norm_path.read_text())
    if "ideal_loss_stretch" not in dataset_norm:
        sys.exit(f"[data] {norm_path} has no `ideal_loss_stretch`. It was "
                 f"probably written by ml-decon's dataset/gen_data.py, whose "
                 f"patches are already asinh-stretched -- this trainer expects "
                 f"linear patches from gen_data_linear.py.")
    print(f"[data] dataset norm ({norm_path}): {dataset_norm}")
    stretch = make_stretch(dataset_norm["ideal_loss_stretch"])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    loss_kwargs = dict(eps_weight=args.eps_weight,
                       stretch_weight=args.stretch_weight)

    # --- model / optim --------------------------------------------------
    model = ConditionalFlatCNN(channels=args.channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ConditionalFlatCNN, {n_params/1e6:.2f}M params, "
          f"dilations {ConditionalFlatCNN.DILATIONS} (41-px receptive field)")
    print(f"[loss] {args.eps_weight} * eps_mse + {args.stretch_weight} * "
          f"stretch_mse (asinh domain, unclamped), null token {NULL_TOKEN}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    ema, ema_is_lib = make_ema(model)
    alpha_bar = cosine_alpha_bar(args.timesteps).to(device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(args.checkpoint_dir / "dataset_norm.json", "w") as f:
        json.dump(dataset_norm, f, indent=2)

    # --- overfit-one-batch sanity check ---------------------------------
    if args.overfit_one_batch:
        ideal, observed = next(iter(train_loader))
        ideal, observed = ideal.to(device), observed.to(device)

        # Deterministic evaluation pack: FROZEN noise + FIXED timestep grid.
        # The per-step training loss resamples t and eps every iteration, so it
        # bounces around as it draws easy (high-t) or hard (low-t) subproblems
        # -- it is NOT a reliable convergence signal on its own.
        gen = torch.Generator(device="cpu").manual_seed(0)
        eval_eps = torch.randn(ideal.shape, generator=gen).to(device)
        eval_ts = [int(f * args.timesteps) for f in (0.05, 0.25, 0.5, 0.75, 0.95)]

        @torch.no_grad()
        def fixed_eval():
            model.eval()
            losses = []
            for ti in eval_ts:
                t = torch.full((ideal.shape[0],), ti, device=device,
                               dtype=torch.long)
                ab = alpha_bar[t][:, None, None, None]
                x_t = ab.sqrt() * ideal + (1 - ab).sqrt() * eval_eps
                losses.append(F.mse_loss(model(x_t, observed, t),
                                         eval_eps).item())
            model.train()
            return float(np.mean(losses)), losses

        # A hotter LR is appropriate for pure memorization.
        sanity_opt = torch.optim.AdamW(model.parameters(), lr=args.sanity_lr)
        print(f"[sanity] Overfitting a single batch of {ideal.shape[0]} pairs "
              f"for {args.sanity_steps} steps (lr {args.sanity_lr}).")
        print("[sanity] Reference points: untrained model scores ~1.0 on the "
              "fixed eval (variance of eps). Healthy memorization: fixed-eval "
              "loss steadily decreasing, reaching <0.05 by the end. The "
              "fixed eval measures the eps term only, so it stays comparable "
              "to the asinh-domain trainer's numbers.")
        model.train()
        running = None
        for step in range(1, args.sanity_steps + 1):
            sanity_opt.zero_grad()
            loss, eps_mse, stretch_mse = diffusion_loss(
                model, ideal, observed, alpha_bar, device, stretch,
                **loss_kwargs)
            loss.backward()
            sanity_opt.step()
            running = (loss.item() if running is None
                       else 0.98 * running + 0.02 * loss.item())
            if step % 100 == 0:
                ev_mean, ev_per_t = fixed_eval()
                per_t = "  ".join(f"t={t}:{l:.3f}"
                                  for t, l in zip(eval_ts, ev_per_t))
                print(f"  step {step:5d}  train(avg) {running:.4f}  "
                      f"[eps {eps_mse.item():.4f} stretch "
                      f"{stretch_mse.item():.4f}]  "
                      f"fixed-eval {ev_mean:.4f}   [{per_t}]")
        ev_mean, _ = fixed_eval()
        verdict = ("PASS" if ev_mean < 0.05 else
                   "MARGINAL -- decreasing but not converged; rerun with "
                   "more --sanity-steps" if ev_mean < 0.2 else
                   "FAIL -- check the data ranges printed above")
        print(f"[sanity] Final fixed-eval loss {ev_mean:.4f}: {verdict}")
        return

    # --- full training loop ---------------------------------------------
    best_val = float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running = np.zeros(3)
        n_seen = 0
        for ideal, observed in train_loader:
            ideal, observed = ideal.to(device), observed.to(device)
            optimizer.zero_grad()
            loss, eps_mse, stretch_mse = diffusion_loss(
                model, ideal, observed, alpha_bar, device, stretch,
                p_uncond=args.p_uncond,
                identity_frac=args.identity_frac,
                identity_t_max=args.identity_t_max,
                low_t_frac=args.low_t_frac,
                low_t_max=args.low_t_max,
                **loss_kwargs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_is_lib:
                ema.update()
            else:
                ema.update(model)
            running += np.array([loss.item(), eps_mse.item(),
                                 stretch_mse.item()]) * ideal.shape[0]
            n_seen += ideal.shape[0]

        train_total, train_eps, train_stretch = running / max(n_seen, 1)
        val_total, val_eps, val_stretch = validate(
            model, val_loader, alpha_bar, device, stretch, **loss_kwargs)
        dt = time.time() - t0
        print(f"[epoch {epoch:3d}/{args.epochs}] "
              f"train {train_total:.5f} (eps {train_eps:.5f} stretch "
              f"{train_stretch:.5f})  "
              f"val {val_total:.5f} (eps {val_eps:.5f} stretch "
              f"{val_stretch:.5f})  ({dt:.1f}s)")

        save_checkpoint(args.checkpoint_dir / "last.pt", model, ema,
                        ema_is_lib, optimizer, epoch, dataset_norm, args)
        if val_total < best_val:
            best_val = val_total
            save_checkpoint(args.checkpoint_dir / "best.pt", model, ema,
                            ema_is_lib, optimizer, epoch, dataset_norm, args)
            print(f"          -> new best val loss, saved best.pt")

    print(f"[done] best val loss {best_val:.5f}. Use the 'ema_state' weights "
          f"from best.pt as the prior in the PnP loop. The model works in the "
          f"dataset's stored LINEAR domain; 'dataset_norm' in the checkpoint is "
          f"the only map back to physical flux, and the unconditional path "
          f"needs conditioning filled with {NULL_TOKEN}, not zeros.")


# Mandatory guard: this file contains a training loop and must never retrain on
# import (e.g. when a solver imports ConditionalFlatCNN).
if __name__ == "__main__":
    main()

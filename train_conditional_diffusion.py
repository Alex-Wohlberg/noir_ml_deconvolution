"""
train_conditional_diffusion.py

Trains a *conditional* DDPM whose denoiser is a flat (no-downsampling) CNN,
learning p(sharp | blurry) from paired astronomical FITS patches.

Why conditional?
----------------
An unconditional prior trained only on sharp patches knows what star fields
look like, but nothing about how a *specific* blurry observation constrains
the reconstruction. By concatenating the blurry counterpart as a conditioning
channel at every diffusion step, the model learns the joint statistics of the
(sharp, blurry) pairing. When later used as the proximal operator / prior in
the PnP loop, this suppresses hallucinated sources: the prior itself has been
taught which sharp structures are consistent with a given blurry input.

Why a flat CNN (not U-Net)?
---------------------------
Sharp patches have sub-pixel spatial autocorrelation and an approximately
flat power spectrum -- there is no multi-scale structure for a U-Net's
encoder/decoder pyramid to exploit, and downsampling would destroy the
point-source statistics we care about. Instead we use a DnCNN-style stack
with a dilation pyramid (1,2,3,4,4,3,2,1) giving a 41-pixel receptive field
on 64x64 patches, with FiLM (scale/shift) timestep conditioning.

Data handling
-------------
* Raw FITS in linear flux units. NO upstream scaling assumed.
* Blurry patches carry a ~21x multiplicative flux excess relative to sharp
  (un-normalized PSF kernel, kernel.sum() ~ 21.84). We correct this in
  LINEAR flux space before any transform. The ratio is auto-estimated from
  the data by default (median of per-pair flux-sum ratios), or can be pinned
  with --flux-ratio. After you regenerate patches with the renormalized
  kernel, run with --flux-ratio 1.0.
* Shared asinh transform  z = asinh(x / b) / A  applied to BOTH images after
  flux correction, matching astro_transforms.py conventions:
      b ~ sigma-clipped noise std of the blurry patches
      A = asinh(p99.9_blurry / b)   so blurry p99.9 maps to z = 1
  The (b, A, flux_ratio) triple is saved in every checkpoint so the PnP
  loop can invert the transform exactly at the linear<->asinh seam.
* Filenames carry 7 augmentation variants per scene
  (_original, _flip_h, _flip_v, _flip_diag, _rot90, _rot180, _rot270).
  Train/val split is done at SCENE level to prevent leakage of augmented
  copies of validation scenes into training.

Usage
-----
    cd ~/noir_ml/mycode
    python train_conditional_diffusion.py \
        --blurry-dir patches/blurry/fits \
        --sharp-dir  patches/sharp/fits \
        --epochs 100 --batch-size 32

    # Sanity check first (memorize a single batch; loss should -> ~0):
    python train_conditional_diffusion.py --overfit-one-batch

Requires: torch, numpy, astropy. Optional: ema_pytorch (falls back to a
built-in EMA if absent).
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from astropy.io import fits
except ImportError:
    sys.exit("astropy is required: pip install astropy")


# ----------------------------------------------------------------------------
# Filename / scene handling
# ----------------------------------------------------------------------------

AUG_SUFFIXES = (
    "_original", "_flip_h", "_flip_v", "_flip_diag",
    "_rot90", "_rot180", "_rot270",
)
_AUG_RE = re.compile(r"(_original|_flip_h|_flip_v|_flip_diag|_rot90|_rot180|_rot270)$")


def scene_id(stem: str) -> str:
    """Strip the augmentation suffix so all 7 variants share one scene id."""
    return _AUG_RE.sub("", stem)


def collect_pairs(blurry_dir: Path, sharp_dir: Path):
    """Match blurry/sharp FITS files by filename; return list of (blurry, sharp)."""
    blurry_files = {p.name: p for p in sorted(blurry_dir.glob("*.fits"))}
    sharp_files = {p.name: p for p in sorted(sharp_dir.glob("*.fits"))}
    common = sorted(set(blurry_files) & set(sharp_files))
    missing_b = set(sharp_files) - set(blurry_files)
    missing_s = set(blurry_files) - set(sharp_files)
    if missing_b or missing_s:
        print(f"[pairing] WARNING: {len(missing_b)} sharp files lack a blurry "
              f"partner, {len(missing_s)} blurry files lack a sharp partner. "
              f"Unmatched files are skipped.")
    if not common:
        sys.exit("[pairing] No matched FITS pairs found. Check the two paths.")
    pairs = [(blurry_files[n], sharp_files[n]) for n in common]
    print(f"[pairing] {len(pairs)} matched pairs "
          f"({len({scene_id(Path(n).stem) for n in common})} unique scenes).")
    return pairs


def scene_split(pairs, val_frac: float, seed: int):
    """Split pairs at scene level so augmented variants never straddle the split."""
    by_scene = defaultdict(list)
    for pair in pairs:
        by_scene[scene_id(pair[0].stem)].append(pair)
    scenes = sorted(by_scene)
    rng = random.Random(seed)
    rng.shuffle(scenes)
    n_val = max(1, int(round(val_frac * len(scenes))))
    val_scenes = set(scenes[:n_val])
    train = [p for s in scenes[n_val:] for p in by_scene[s]]
    val = [p for s in val_scenes for p in by_scene[s]]
    print(f"[split] {len(scenes) - n_val} train scenes ({len(train)} patches) / "
          f"{n_val} val scenes ({len(val)} patches)")
    return train, val


# ----------------------------------------------------------------------------
# FITS I/O and normalization statistics
# ----------------------------------------------------------------------------

def load_fits(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False) as hdul:
        data = hdul[0].data
        if data is None:  # data sometimes lives in the first extension
            data = hdul[1].data
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        arr = np.squeeze(arr)
    if not np.all(np.isfinite(arr)):
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    return arr


def sigma_clipped_std(x: np.ndarray, sigma: float = 3.0, iters: int = 5) -> float:
    """Robust noise-floor estimate (same spirit as _sigma_clipped_std in the
    PnP code). Iteratively clips outliers, returns std of the residual core."""
    data = x.ravel().astype(np.float64)
    for _ in range(iters):
        med = np.median(data)
        std = data.std()
        if std == 0:
            break
        keep = np.abs(data - med) < sigma * std
        if keep.sum() == data.size:
            break
        data = data[keep]
    return float(max(data.std(), 1e-12))


def estimate_stats(train_pairs, flux_ratio_arg, n_sample=256, seed=0):
    """Single pass over a random subsample of TRAINING pairs to estimate:
       * flux_ratio : median( sum(blurry) / sum(sharp) )  -- linear space
       * b          : sigma-clipped noise std of flux-corrected blurry patches
       * A          : asinh(p99.9 / b) of flux-corrected blurry patches
    Percentile-based throughout: raw FITS contain extreme outlier pixels
    (max up to ~2450x the 99.9th percentile), so min/max stats are useless.
    """
    rng = random.Random(seed)
    sample = rng.sample(train_pairs, min(n_sample, len(train_pairs)))

    sums_sharp, sums_blur, noise_stds, blurry_vals = [], [], [], []
    n_pix = None
    for b_path, s_path in sample:
        blur = load_fits(b_path)
        sharp = load_fits(s_path)
        n_pix = blur.size
        s_sum = float(sharp.sum())
        if s_sum > 0:
            sums_sharp.append(s_sum)
            sums_blur.append(float(blur.sum()))
        noise_stds.append(sigma_clipped_std(blur))
        blurry_vals.append(blur.ravel())

    sums_sharp = np.asarray(sums_sharp)
    sums_blur = np.asarray(sums_blur)

    # Separate multiplicative from additive via robust regression:
    #     blur_sum = slope * sharp_sum + intercept
    #   slope       -> multiplicative factor (kernel normalization error)
    #   intercept/N -> additive per-pixel pedestal (sky background)
    # Why not median-of-ratios or median background subtraction? Ratios
    # conflate the two effects, and per-patch median subtraction
    # OVER-subtracts in crowded fields (M31): overlapping PSF wings build a
    # quasi-uniform floor of real source flux that the median mistakes for
    # sky. The regression needs no crowding assumption -- patches with more
    # stars carry proportionally more smeared flux, and that co-variation
    # is exactly what the slope captures.
    # Theil-Sen: median of pairwise slopes, robust to outlier patches.
    ii, jj = np.triu_indices(len(sums_sharp), k=1)
    dx = sums_sharp[jj] - sums_sharp[ii]
    ok = np.abs(dx) > 0
    slope = float(np.median((sums_blur[jj][ok] - sums_blur[ii][ok]) / dx[ok]))
    intercept = float(np.median(sums_blur - slope * sums_sharp))
    pedestal_per_px = intercept / n_pix
    med_ratio = float(np.median(sums_blur / sums_sharp))

    if flux_ratio_arg is not None:
        flux_ratio = float(flux_ratio_arg)
        print(f"[stats] flux ratio: using user-supplied {flux_ratio:.4f} "
              f"(regression slope was {slope:.4f})")
    else:
        flux_ratio = slope
        print(f"[stats] flux model (Theil-Sen over {len(sums_sharp)} pairs): "
              f"blur_sum = {slope:.4f} * sharp_sum + {intercept:.6g}")
        print(f"[stats]   multiplicative factor (slope): {slope:.4f}   "
              f"[median-of-ratios would give {med_ratio:.4f} -- inflated by "
              f"the pedestal]")
        print(f"[stats]   additive pedestal: {pedestal_per_px:.6g} per pixel "
              f"({intercept:.6g} per {n_pix}-px patch)")
        rel = abs(slope - 21.84) / 21.84
        if rel < 0.05:
            print("[stats]   -> slope matches the un-normalized PSF kernel "
                  "(kernel.sum() ~ 21.84); the raw-ratio excess was just "
                  "the sky pedestal. Kernel diagnosis confirmed.")
        else:
            print(f"[stats]   -> NOTE: slope is {100*rel:.1f}% away from "
                  f"the kernel sum 21.84. The multiplicative discrepancy "
                  f"is NOT fully explained by kernel normalization -- "
                  f"bring this to the advisor alongside the zero-floor "
                  f"question before trusting downstream photometry.")

    all_blur = np.concatenate(blurry_vals) / flux_ratio  # corrected linear flux
    b = float(np.median([s / flux_ratio for s in noise_stds]))
    p999 = float(np.percentile(all_blur, 99.9))
    A = float(np.arcsinh(p999 / b)) if p999 > 0 else 1.0
    print(f"[stats] asinh params: b = {b:.6g} (noise floor), "
          f"A = {A:.4f} (from blurry p99.9 = {p999:.6g})")
    return {"flux_ratio": flux_ratio, "b": b, "A": A, "p999_blurry": p999}


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------

class PairedAsinhDataset(Dataset):
    """Yields (sharp_z, blurry_z) tensors in the shared asinh domain.
    All flux correction happens in LINEAR space before the transform."""

    def __init__(self, pairs, stats):
        self.pairs = pairs
        self.flux_ratio = stats["flux_ratio"]
        self.b = stats["b"]
        self.A = stats["A"]

    def __len__(self):
        return len(self.pairs)

    def asinh_fwd(self, x: np.ndarray) -> np.ndarray:
        return np.arcsinh(x / self.b) / self.A

    def __getitem__(self, idx):
        b_path, s_path = self.pairs[idx]
        blur = load_fits(b_path) / self.flux_ratio   # linear-space correction
        sharp = load_fits(s_path)                    # sharp is the reference
        blur_z = self.asinh_fwd(blur)
        sharp_z = self.asinh_fwd(sharp)
        return (
            torch.from_numpy(sharp_z).unsqueeze(0).float(),
            torch.from_numpy(blur_z).unsqueeze(0).float(),
        )


# ----------------------------------------------------------------------------
# Cosine noise schedule (Nichol & Dhariwal 2021, as used in the DPS paper)
# ----------------------------------------------------------------------------

def cosine_alpha_bar(T: int, s: float = 0.008) -> torch.Tensor:
    steps = torch.arange(T + 1, dtype=torch.float64)
    f = torch.cos(((steps / T) + s) / (1 + s) * math.pi / 2) ** 2
    alpha_bar = f / f[0]
    betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
    betas = betas.clamp(1e-8, 0.999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0).float()  # alpha_bar[t], t = 0..T-1


# ----------------------------------------------------------------------------
# Model: flat CNN denoiser with FiLM timestep conditioning
# ----------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / half
        )
        args = t.float()[:, None] * freqs[None, :]
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

    Input channels: [x_t (noisy sharp), y (clean blurry conditioning)].
    The blurry channel is NEVER noised -- it is a fixed conditioning signal
    at every timestep, which is what injects the pairing statistics.
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

def diffusion_loss(model, sharp, blurry, alpha_bar, device, p_uncond=0.0):
    """eps-prediction DDPM loss, conditioned on blurry.

    p_uncond > 0 enables CLASSIFIER-FREE GUIDANCE training: that fraction of
    the batch has its conditioning channel replaced by the NULL token (all
    zeros), so the same weights learn BOTH
        eps_theta(x_t, y, t)     the conditional/posterior score, and
        eps_theta(x_t, 0, t)     a genuine UNCONDITIONAL prior score.

    Why this matters for the PnP use: RED / Graikos regularizers assume the
    denoiser models the PRIOR p(x). A purely conditional denoiser gives the
    POSTERIOR score, which already contains the likelihood -- using it next
    to an explicit ||y - Ax||^2 term double-counts the measurement. With
    condition dropout you can call the model with the null token to get the
    clean prior score, and keep the conditional path for sampling/anchoring.

    All-zeros is a safe null token here: real blurry patches are never
    identically zero, so it is unambiguously distinguishable.
    """
    bsz = sharp.shape[0]
    T = alpha_bar.shape[0]
    t = torch.randint(0, T, (bsz,), device=device)
    ab = alpha_bar[t][:, None, None, None]
    eps = torch.randn_like(sharp)
    x_t = ab.sqrt() * sharp + (1 - ab).sqrt() * eps
    if p_uncond > 0:
        drop = (torch.rand(bsz, device=device) < p_uncond)
        blurry = torch.where(drop[:, None, None, None],
                             torch.zeros_like(blurry), blurry)
    eps_pred = model(x_t, blurry, t)
    return F.mse_loss(eps_pred, eps)


@torch.no_grad()
def validate(model, loader, alpha_bar, device):
    model.eval()
    total, n = 0.0, 0
    for sharp, blurry in loader:
        sharp, blurry = sharp.to(device), blurry.to(device)
        loss = diffusion_loss(model, sharp, blurry, alpha_bar, device)
        total += loss.item() * sharp.shape[0]
        n += sharp.shape[0]
    model.train()
    return total / max(n, 1)


def save_checkpoint(path, model, ema, ema_is_lib, optimizer, epoch, stats, args):
    ckpt = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "ema_state": (ema.ema_model.state_dict() if ema_is_lib
                      else ema.state_dict()),
        "optimizer_state": optimizer.state_dict(),
        # The transform triple: REQUIRED by the PnP loop to cross the
        # linear<->asinh seam correctly. Checksum-style redundancy on purpose.
        "transform": {
            "flux_ratio": stats["flux_ratio"],
            "asinh_b": stats["b"],
            "asinh_A": stats["A"],
        },
        "arch": {"channels": args.channels,
                 "dilations": list(ConditionalFlatCNN.DILATIONS),
                 "t_dim": 128, "in_channels": 2},
        "diffusion": {"timesteps": args.timesteps, "schedule": "cosine"},
        # p_uncond > 0 means model(x_t, zeros, t) is a valid UNCONDITIONAL
        # prior score (classifier-free guidance); consumers should check this
        # before using the null-token path.
        "p_uncond": args.p_uncond,
        "args": vars(args),
    }
    torch.save(ckpt, path)


def main():
    ap = argparse.ArgumentParser(
        description="Train a conditional flat-CNN DDPM on sharp/blurry FITS pairs.")
    ap.add_argument("--blurry-dir", type=Path,
                    default=Path("/home/alex/noir_ml/mycode/patches/blurry/fits"))
    ap.add_argument("--sharp-dir", type=Path,
                    default=Path("/home/alex/noir_ml/mycode/patches/sharp/fits"))
    ap.add_argument("--checkpoint-dir", type=Path,
                    default=Path("checkpoints_cond_diffusion"))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--p-uncond", type=float, default=0.15,
                    help="classifier-free-guidance dropout: fraction of "
                         "training samples whose blurry conditioning is "
                         "replaced by the null token (zeros). >0 makes the "
                         "SAME weights usable as an unconditional prior via "
                         "model(x_t, zeros, t) -- required if you want to "
                         "use this model in a RED/Graikos regularizer "
                         "alongside an explicit ||y-Ax||^2 term without "
                         "double-counting the likelihood. 0 = purely "
                         "conditional (old behaviour).")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--flux-ratio", type=float, default=None,
                    help="Pin the blurry/sharp flux ratio (e.g. 21.84). "
                         "Default: auto-estimate from data. Use 1.0 after "
                         "patches are regenerated with a normalized kernel.")
    ap.add_argument("--stats-sample", type=int, default=256,
                    help="Number of pairs sampled for normalization stats.")
    ap.add_argument("--overfit-one-batch", action="store_true",
                    help="Sanity check: memorize a single batch and track a "
                         "deterministic fixed-noise eval loss.")
    ap.add_argument("--sanity-steps", type=int, default=3000,
                    help="Steps for the overfit-one-batch check.")
    ap.add_argument("--sanity-lr", type=float, default=3e-4,
                    help="Learning rate for the overfit-one-batch check "
                         "(hotter than the training default, appropriate "
                         "for pure memorization).")
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
    pairs = collect_pairs(args.blurry_dir, args.sharp_dir)
    train_pairs, val_pairs = scene_split(pairs, args.val_frac, args.seed)
    stats = estimate_stats(train_pairs, args.flux_ratio,
                           n_sample=args.stats_sample, seed=args.seed)

    train_ds = PairedAsinhDataset(train_pairs, stats)
    val_ds = PairedAsinhDataset(val_pairs, stats)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"),
                              drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers)

    # --- model / optim --------------------------------------------------
    model = ConditionalFlatCNN(channels=args.channels).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] ConditionalFlatCNN, {n_params/1e6:.2f}M params, "
          f"dilations {ConditionalFlatCNN.DILATIONS} (41-px receptive field)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-5)
    ema, ema_is_lib = make_ema(model)
    alpha_bar = cosine_alpha_bar(args.timesteps).to(device)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    with open(args.checkpoint_dir / "transform.json", "w") as f:
        json.dump({"flux_ratio": stats["flux_ratio"],
                   "asinh_b": stats["b"], "asinh_A": stats["A"]}, f, indent=2)

    # --- overfit-one-batch sanity check ---------------------------------
    if args.overfit_one_batch:
        sharp, blurry = next(iter(train_loader))
        sharp, blurry = sharp.to(device), blurry.to(device)

        # Deterministic evaluation pack: FROZEN noise + FIXED timestep grid.
        # The per-step training loss resamples t and eps every iteration, so
        # it bounces around as it draws easy (high-t) or hard (low-t)
        # subproblems -- it is NOT a reliable convergence signal on its own.
        # This eval number is comparable across steps.
        gen = torch.Generator(device="cpu").manual_seed(0)
        eval_eps = torch.randn(sharp.shape, generator=gen).to(device)
        eval_ts = [int(f * args.timesteps) for f in (0.05, 0.25, 0.5, 0.75, 0.95)]

        @torch.no_grad()
        def fixed_eval():
            model.eval()
            losses = []
            for ti in eval_ts:
                t = torch.full((sharp.shape[0],), ti, device=device,
                               dtype=torch.long)
                ab = alpha_bar[t][:, None, None, None]
                x_t = ab.sqrt() * sharp + (1 - ab).sqrt() * eval_eps
                losses.append(F.mse_loss(model(x_t, blurry, t),
                                         eval_eps).item())
            model.train()
            return float(np.mean(losses)), losses

        # A hotter LR is appropriate for pure memorization.
        sanity_opt = torch.optim.AdamW(model.parameters(), lr=args.sanity_lr)
        print(f"[sanity] Overfitting a single batch of {sharp.shape[0]} "
              f"pairs for {args.sanity_steps} steps (lr {args.sanity_lr}).")
        print("[sanity] Reference points: untrained model scores ~1.0 "
              "(variance of eps). Healthy memorization: fixed-eval loss "
              "steadily decreasing, reaching <0.05 by the end. The low-t "
              "entries of the per-t breakdown are the hardest and fall last.")
        model.train()
        running = None
        for step in range(1, args.sanity_steps + 1):
            sanity_opt.zero_grad()
            loss = diffusion_loss(model, sharp, blurry, alpha_bar, device)
            loss.backward()
            sanity_opt.step()
            running = (loss.item() if running is None
                       else 0.98 * running + 0.02 * loss.item())
            if step % 100 == 0:
                ev_mean, ev_per_t = fixed_eval()
                per_t = "  ".join(f"t={t}:{l:.3f}"
                                  for t, l in zip(eval_ts, ev_per_t))
                print(f"  step {step:5d}  train(avg) {running:.4f}  "
                      f"fixed-eval {ev_mean:.4f}   [{per_t}]")
        ev_mean, _ = fixed_eval()
        verdict = ("PASS" if ev_mean < 0.05 else
                   "MARGINAL -- decreasing but not converged; rerun with "
                   "more --sanity-steps" if ev_mean < 0.2 else
                   "FAIL -- check the normalization stats printed above")
        print(f"[sanity] Final fixed-eval loss {ev_mean:.4f}: {verdict}")
        return

    # --- full training loop ---------------------------------------------
    best_val = float("inf")
    model.train()
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        running, n_seen = 0.0, 0
        for sharp, blurry in train_loader:
            sharp, blurry = sharp.to(device), blurry.to(device)
            optimizer.zero_grad()
            loss = diffusion_loss(model, sharp, blurry, alpha_bar, device,
                                  p_uncond=args.p_uncond)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if ema_is_lib:
                ema.update()
            else:
                ema.update(model)
            running += loss.item() * sharp.shape[0]
            n_seen += sharp.shape[0]

        train_loss = running / max(n_seen, 1)
        val_loss = validate(model, val_loader, alpha_bar, device)
        dt = time.time() - t0
        print(f"[epoch {epoch:3d}/{args.epochs}] train {train_loss:.5f}  "
              f"val {val_loss:.5f}  ({dt:.1f}s)")

        save_checkpoint(args.checkpoint_dir / "last.pt", model, ema,
                        ema_is_lib, optimizer, epoch, stats, args)
        if val_loss < best_val:
            best_val = val_loss
            save_checkpoint(args.checkpoint_dir / "best.pt", model, ema,
                            ema_is_lib, optimizer, epoch, stats, args)
            print(f"          -> new best val loss, saved best.pt")

    print(f"[done] best val loss {best_val:.5f}. Use the 'ema_state' weights "
          f"from best.pt as the prior in the PnP loop, and read the "
          f"'transform' dict for the linear<->asinh seam constants.")


# Mandatory guard: this file contains a training loop and must never
# retrain on import (e.g. when pnp_deconvolve.py imports ConditionalFlatCNN).
if __name__ == "__main__":
    main()
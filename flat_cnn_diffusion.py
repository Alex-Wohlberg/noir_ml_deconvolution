"""
flat_cnn_diffusion.py

A "flat" (no downsampling) CNN diffusion model for noise prediction,
intended for sparse point-source imagery (deconvolved star fields) where
there is little multi-scale structure for a U-Net to exploit.

Architecture: DnCNN-style residual stack of 3x3 convs at full resolution.
The receptive field is grown with a dilation pyramid instead of pooling,
so sparse bright pixels are never diluted by downsampling. Timestep
conditioning is injected into every block via FiLM (scale + shift derived
from a sinusoidal embedding), matching the conditioning style of the
ddpm_mnist.py scaffold.

Works on two datasets:
  --dataset mnist   : torchvision MNIST, padded 28 -> 32, mapped to [-1, 1]
  --dataset stars   : directory of 64x64 FITS patches (or a single .npy
                      stack), normalized with global SKY/SCALE constants
                      and clipped, then mapped to [-1, 1]

Usage examples:
  # sanity check first (should drive loss to ~0 on one batch)
  python flat_cnn_diffusion.py --dataset mnist --overfit-one-batch

  # full MNIST run
  python flat_cnn_diffusion.py --dataset mnist --epochs 20

  # star patches (FITS directory)
  python flat_cnn_diffusion.py --dataset stars \
      --data-dir /home/alex/noir_ml/mycode/patches/sharp/fits \
      --epochs 100 --out flat_cnn_stars.pt

  # sample from a trained checkpoint
  python flat_cnn_diffusion.py --dataset stars --data-dir ... \
      --sample-only --ckpt flat_cnn_stars.pt
"""

import argparse
import glob
import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset

# ----------------------------------------------------------------------------
# Star-patch normalization now lives in astro_transforms.AsinhTransform:
#     z = asinh(x / b) / A        b ~ background noise sigma (linear regime)
#     x = b * sinh(z * A)         A from blurry 99.9th pct -> z = 1.0
# Derived from data at train time, frozen into the checkpoint, and asserted
# against at inference. No lower clip at zero: negative background noise
# passes through (asinh is odd), preserving zero-mean noise statistics.
# MNIST keeps its own simple [-1, 1] normalization.
# ----------------------------------------------------------------------------
from astro_transforms import AsinhTransform


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================================
# Timestep embedding
# ============================================================================
def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """t: (B,) integer timesteps -> (B, dim) embedding."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    args = t.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ============================================================================
# Model: flat dilated-conv residual stack with FiLM timestep conditioning
# ============================================================================
class FiLMBlock(nn.Module):
    """3x3 conv (dilated) -> GroupNorm -> FiLM(t) -> SiLU, with residual add.

    No spatial resizing anywhere: input and output are (B, C, H, W).
    """

    def __init__(self, channels: int, dilation: int, t_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=3, padding=dilation, dilation=dilation
        )
        self.norm = nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)
        # FiLM: per-channel scale and shift from the timestep embedding
        self.film = nn.Linear(t_dim, 2 * channels)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv(x)
        h = self.norm(h)
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        h = h * (1 + scale[:, :, None, None]) + shift[:, :, None, None]
        h = F.silu(h)
        return x + h


class FlatCNN(nn.Module):
    """Noise-prediction network eps_theta(x_t, t) with no down/upsampling.

    Receptive field with the default dilation pattern [1,2,3,4,4,3,2,1]:
      RF = 1 + 2 * sum(dilations) = 41 pixels
    """

    def __init__(
        self,
        in_channels: int = 1,
        base: int = 64,
        dilations=(1, 2, 3, 4, 4, 3, 2, 1),
        t_dim: int = 128,
    ):
        super().__init__()
        self.t_dim = t_dim
        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim * 2), nn.SiLU(), nn.Linear(t_dim * 2, t_dim)
        )
        self.head = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [FiLMBlock(base, d, t_dim) for d in dilations]
        )
        self.tail = nn.Conv2d(base, in_channels, kernel_size=3, padding=1)
        # Zero-init the tail so the model starts by predicting zero noise;
        # stabilizes early training.
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t_emb = self.t_mlp(sinusoidal_embedding(t, self.t_dim))
        h = self.head(x)
        for block in self.blocks:
            h = block(h, t_emb)
        return self.tail(h)  # predicted noise, same shape as x


# Diffusion schedule
class Schedule:
    def __init__(self, timesteps: int = 300, beta_start: float = 1e-4,
                 beta_end: float = 0.02, device: torch.device = torch.device("cpu")):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """Forward process: x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps."""
        ab = self.alpha_bars[t][:, None, None, None]
        return ab.sqrt() * x0 + (1 - ab).sqrt() * noise

    @torch.no_grad()
    def p_sample_loop(self, model: nn.Module, shape, device: torch.device) -> torch.Tensor:
        """Full reverse process from pure noise (ancestral DDPM sampling)."""
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = model(x, t)
            alpha = self.alphas[i]
            ab = self.alpha_bars[i]
            mean = (x - (1 - alpha) / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                x = mean + self.betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x

    @torch.no_grad()
    def denoise_from(self, model: nn.Module, x_t: torch.Tensor, t_start: int) -> torch.Tensor:
        """Iterative reverse diffusion starting from a given x_t at t_start.

        This is the hook the PnP-PGD proximal step will use: noise the
        current iterate to t_start, then run the reverse chain back to 0.
        """
        x = x_t.clone()
        for i in reversed(range(t_start + 1)):
            t = torch.full((x.shape[0],), i, device=x.device, dtype=torch.long)
            eps = model(x, t)
            alpha = self.alphas[i]
            ab = self.alpha_bars[i]
            mean = (x - (1 - alpha) / (1 - ab).sqrt() * eps) / alpha.sqrt()
            if i > 0:
                x = mean + self.betas[i].sqrt() * torch.randn_like(x)
            else:
                x = mean
        return x

    @torch.no_grad()
    def denoise_multistep(self, model: nn.Module, x_t: torch.Tensor,
                          t_start: int, n_steps: int = 4,
                          eta: float = 0.0) -> torch.Tensor:
        """Strided DDIM reverse from t_start down to 0 in n_steps jumps.

        A stronger manifold projection than the one-step Tweedie estimate
        (denoise_x0) without the cost of the full chain (denoise_from): the
        few intermediate re-noise/denoise steps let the model correct its
        own x0 estimate, which concentrates blurry point-source blobs toward
        the sharp delta-like structure the prior was trained on. eta=0 is
        deterministic DDIM (a clean, repeatable prox); eta>0 injects
        ancestral noise. Cost: n_steps forward passes per prox call.
        """
        if t_start <= 0:
            return self.denoise_x0(model, x_t, 0)
        # respaced grid t_start -> 0 inclusive (unique, descending)
        grid = np.unique(np.linspace(0, t_start, n_steps + 1).astype(int))[::-1]
        x = x_t
        for j in range(len(grid)):
            t_cur = int(grid[j])
            t_nxt = int(grid[j + 1]) if j + 1 < len(grid) else -1
            tt = torch.full((x.shape[0],), t_cur, device=x.device, dtype=torch.long)
            eps = model(x, tt)
            ab_t = self.alpha_bars[t_cur]
            x0 = (x - (1 - ab_t).sqrt() * eps) / ab_t.sqrt()
            if t_nxt < 0:
                x = x0
                break
            ab_n = self.alpha_bars[t_nxt]
            sigma = (eta * ((1 - ab_n) / (1 - ab_t)).sqrt()
                     * (1 - ab_t / ab_n).sqrt())
            dir_coeff = (1 - ab_n - sigma ** 2).clamp(min=0).sqrt()
            x = ab_n.sqrt() * x0 + dir_coeff * eps
            if eta > 0:
                x = x + sigma * torch.randn_like(x)
        return x

    def sigma_to_t(self, sigma_model: float) -> int:
        """Timestep whose marginal noise level best matches sigma_model.

        DDPM marginal: x_t = sqrt(abar_t) x0 + sqrt(1 - abar_t) eps, so the
        effective noise std relative to signal is
            sigma(t) = sqrt(1 - abar_t) / sqrt(abar_t).
        IMPORTANT: sigma_model must be in MODEL space (the space the prior
        trained in) -- convert a flux-space residual level with
        AsinhTransform.sigma_flux_to_model first.
        """
        sigmas = ((1 - self.alpha_bars) / self.alpha_bars).sqrt()
        return int(torch.argmin((sigmas - sigma_model).abs()).item())

    @torch.no_grad()
    def denoise_x0(self, model: nn.Module, x_t: torch.Tensor, t: int) -> torch.Tensor:
        """One-step x0 estimate via Tweedie's formula:

            x0_hat = (x_t - sqrt(1 - abar_t) * eps_hat) / sqrt(abar_t)

        Used by AsinhProx: cheaper and less hallucination-prone than running
        the full reverse chain (denoise_from) from t back to 0.
        """
        tt = torch.full((x_t.shape[0],), t, device=x_t.device, dtype=torch.long)
        eps_hat = model(x_t, tt)
        ab = self.alpha_bars[t]
        return (x_t - (1 - ab).sqrt() * eps_hat) / ab.sqrt()


class AsinhProx:
    """PnP-PGD proximal operator: flux space in, flux space out.

    Wraps the diffusion prior (trained in MODEL space: asinh z, plus the
    net-space affine x = z/net_scale - 1 when the transform carries one)
    so the PGD loop never touches that space directly:

        x_flux -> to_net -> q_sample to t(sigma) -> denoise -> net_to_flux

    The gradient/data-fidelity step stays in linear flux space (where the
    PSF is linear and the noise is Gaussian); this class is the adapter at
    the prox boundary. sigma_flux is the residual noise level you want the
    prior to remove this iteration -- typically decreasing over PGD iters.
    """

    def __init__(self, model: nn.Module, schedule: Schedule,
                 transform: AsinhTransform, n_steps: int = 1,
                 eta: float = 0.0):
        self.model = model
        self.schedule = schedule
        self.transform = transform
        # n_steps == 1: one-step Tweedie (denoise_x0), cheap but a weak
        # projection that leaves blurry blobs. n_steps > 1: strided DDIM
        # (denoise_multistep), a stronger projection that sharpens point
        # sources -- n_steps forward passes per prox call.
        self.n_steps = int(n_steps)
        self.eta = float(eta)

    @torch.no_grad()
    def __call__(self, x_flux: torch.Tensor, sigma_flux: float) -> torch.Tensor:
        x = self.transform.to_net(x_flux)
        sigma_model = self.transform.sigma_flux_to_model(sigma_flux)
        t = self.schedule.sigma_to_t(sigma_model)
        if t > 0:
            abar = self.schedule.alpha_bars[t]
            noise = torch.randn_like(x)
            x_t = abar.sqrt() * x + (1 - abar).sqrt() * noise  # on-distribution
            if self.n_steps > 1:
                x = self.schedule.denoise_multistep(
                    self.model, x_t, t, n_steps=self.n_steps, eta=self.eta)
            else:
                x = self.schedule.denoise_x0(self.model, x_t, t)

        # Clamp to the trained range before inverting: sinh explodes beyond
        # it, and the denoiser output is not intrinsically bounded. Values
        # outside [net_floor, net_ceil] are outside the training
        # distribution anyway -- pinning them is the correct prior behavior.
        #x = x.clamp(self.transform.net_floor, self.transform.net_ceil)
        return self.transform.net_to_flux(x)


def load_mnist(image_size: int = 32) -> Dataset:
    from torchvision import datasets, transforms

    pad = (image_size - 28) // 2
    tfm = transforms.Compose(
        [
            transforms.Pad(pad),
            transforms.ToTensor(),                      # [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5]),  # [-1, 1]
        ]
    )
    return datasets.MNIST(root="./mnist_data", train=True, download=True, transform=tfm)


class StarPatchDataset(Dataset):
    """64x64 sharp patches from FITS files (or one .npy stack), in asinh space.

    The prior trains on SHARP patches, but the transform constants come from
    BOTH domains: b from the sharp background noise sigma, A from the blurry
    99.9th percentile -- because the PnP loop pushes blurry-domain values
    through the same transform, so the trained range must cover it.

    Assumes the blurry/sharp flux inconsistency (diagnostics plot 07) has
    been corrected upstream: both domains at the same flux scale.
    """

    def __init__(self, sharp_dir: str, blurry_dir: str = None,
                 transform: AsinhTransform = None, image_size: int = 64,
                 b_factor: float = 1.5, pct: float = 99.9,
                 derive_sample: int = 4000, max_patches: int = None,
                 net_scale: float = 0.20, z_ceil: float = 1.45):
        import gc

        self.image_size = image_size

        if transform is not None:
            self.transform = transform
            if self.transform.net_scale is None:
                # legacy JSON without the affine: adopt the requested scale
                self.transform.net_scale = float(net_scale)
        else:
            if blurry_dir is None:
                raise ValueError(
                    "Provide either a prebuilt AsinhTransform or --blurry-dir "
                    "to derive one (A must come from the blurry domain)."
                )
            # Constants are two scalars (a percentile and a noise sigma);
            # a random subsample of patches estimates them to plenty of
            # precision without loading gigabytes into RAM.
            blurry_s = self._load_stack(blurry_dir, image_size,
                                        max_files=derive_sample, seed=0)
            sharp_s = self._load_stack(sharp_dir, image_size,
                                       max_files=derive_sample, seed=0)
            self.transform = AsinhTransform.derive(
                sharp_s, blurry_s, b_factor=b_factor, pct=pct,
                net_scale=net_scale, z_ceil=z_ceil
            )
            del blurry_s, sharp_s
            gc.collect()
        print(f"Normalization: {self.transform}")

        sharp = self._load_stack(sharp_dir, image_size,
                                 max_files=max_patches, seed=1)
        if max_patches is not None:
            print(f"[memory guard] training on a {max_patches}-patch "
                  f"subsample; use the full set on the cluster")
        z = self.transform.to_z(sharp)
        del sharp
        gc.collect()
        lo_z, hi_z = float(z.min()), float(z.max())
        frac_signal = float((z > 0.1).mean())
        if hi_z - lo_z < 0.3:
            raise RuntimeError(
                "Post-transform dynamic range is tiny -- constants look wrong "
                "(did you derive them from flux-inconsistent data?)"
            )

        # Net-space affine: the network must NOT train on raw z. Raw-z data
        # has std ~0.09 against unit diffusion noise, and the prior then
        # collapses to predicting blank background (the flat_cnn_stars.pt
        # failure mode). See AsinhTransform.net_scale.
        if self.transform.net_scale is None:
            raise RuntimeError(
                "transform has no net_scale -- training on raw z is exactly "
                "what made the previous prior collapse to noise. Pass "
                "--net-scale (default 0.20) or use a transform JSON that "
                "stores one.")
        x = self.transform.z_to_net(z)
        del z
        gc.collect()
        self.data = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)
                                     )[:, None, :, :]
        del x
        gc.collect()

        std = self.data.std().item()
        print(f"Loaded {self.data.shape[0]} sharp patches; z range "
              f"[{lo_z:.3f}, {hi_z:.3f}]; frac(z > 0.1) = {frac_signal:.3f}")
        print(f"Net space (x = z/{self.transform.net_scale:.3g} - 1): "
              f"range [{self.data.min().item():.2f}, "
              f"{self.data.max().item():.2f}], std {std:.3f} "
              f"(want >~ 0.3; raw z was ~0.09)")

    @staticmethod
    def _load_stack(data_dir: str, image_size: int,
                    max_files: int = None, seed: int = None) -> np.ndarray:
        """Load patches as float32. If max_files is set, load a random
        subsample of the files (reproducible via seed)."""
        npy_files = sorted(glob.glob(os.path.join(data_dir, "*.npy")))
        fits_files = sorted(glob.glob(os.path.join(data_dir, "*.fits")))
        files = npy_files if (npy_files and not fits_files) else fits_files
        if not files:
            raise FileNotFoundError(f"No .fits or .npy files found in {data_dir}")

        if max_files is not None and len(files) > max_files:
            rng = np.random.default_rng(seed)
            files = [files[i] for i in
                     rng.choice(len(files), size=max_files, replace=False)]

        use_npy = bool(npy_files and not fits_files)
        if not use_npy:
            from astropy.io import fits as afits

        arrays = []
        for f in files:
            data = (np.load(f) if use_npy
                    else afits.getdata(f)).astype(np.float32)
            if data.ndim == 2:
                data = data[None]
            arrays.append(data)

        stack = np.concatenate(arrays, axis=0)
        arrays.clear()
        if stack.shape[-1] != image_size or stack.shape[-2] != image_size:
            raise ValueError(
                f"Expected {image_size}x{image_size} patches, got {stack.shape[-2:]}"
            )
        return stack

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], 0  # dummy label to match MNIST's (img, label)



# Training

# ----------------------------------------------------------------------------
# Training machinery for long runs
# ----------------------------------------------------------------------------
class EMA:
    """Exponential moving average of weights.

    Standard practice for diffusion models and usually the single largest
    quality gain on a long run: the raw SGD iterate keeps rattling around
    the optimum, while the average settles into it. Sampling/denoising
    quality from EMA weights is typically much better than from the live
    weights, and the gap grows with training length.
    """

    def __init__(self, model, decay=0.9999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone()
                        for k, v in model.state_dict().items()
                        if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model, step=None):
        # warm up the decay so early averages are not dominated by the
        # random init: d = min(decay, (1+step)/(10+step))
        d = self.decay
        if step is not None:
            d = min(d, (1.0 + step) / (10.0 + step))
        for k, v in model.state_dict().items():
            if k in self.shadow:
                self.shadow[k].mul_(d).add_(v.detach().float(), alpha=1.0 - d)
            else:
                self.buffers[k] = v.detach().clone()

    def state_dict(self):
        out = {k: v.clone() for k, v in self.shadow.items()}
        out.update({k: v.clone() for k, v in self.buffers.items()})
        return out


def lr_at(step, total_steps, base_lr, warmup, schedule="cosine"):
    """Linear warmup then cosine decay to 5% of base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / max(warmup, 1)
    if schedule != "cosine":
        return base_lr
    prog = (step - warmup) / max(total_steps - warmup, 1)
    prog = min(max(prog, 0.0), 1.0)
    return base_lr * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * prog)))


# ----------------------------------------------------------------------------
# PnP-oriented training objective
# ----------------------------------------------------------------------------
def sample_timesteps(n: int, T: int, low_frac: float, low_max: int,
                     device) -> torch.Tensor:
    """Mixture t-sampling: with prob low_frac draw t ~ U[0, low_max), else
    t ~ U[0, T). Plain uniform sampling spends only ~25% of steps in the
    t<=75 regime where the PnP prox actually operates; the low-t component
    concentrates training there while the uniform component keeps the full
    range covered (the model must still handle every temperature)."""
    t_all = torch.randint(0, T, (n,), device=device)
    t_low = torch.randint(0, max(low_max, 1), (n,), device=device)
    pick = torch.rand(n, device=device) < low_frac
    return torch.where(pick, t_low, t_all)


def diffusion_loss(model, schedule, x0, args, net_ceil: float):
    """eps-prediction MSE with two PnP-oriented modifications.

    identity_frac: that fraction of the batch gets noise = 0, so
      x_t = sqrt(abar)*x0 and the target eps is exactly ZERO. This trains
      the fixed-point property the PnP regularizer needs: applied to a
      clean star field, the denoiser should do (almost) nothing, so that
      x - D(x) measures distance from the star-field manifold instead of
      dragging on-manifold images around (the blob mechanism: the one-step
      posterior mean at high t erased point sources).

    bright_weight: per-pixel loss weight ramping 1 -> 1+bright_weight from
      background (net -1) to the ceiling. Counteracts the asinh compression
      at the bright end (a whole flux DECADE occupies ~25% of net range),
      which otherwise makes peak errors nearly free -- the peak-suppression
      half of the blobbing.
    """
    device = x0.device
    n = x0.shape[0]
    t = sample_timesteps(n, schedule.timesteps, args.low_t_frac,
                         args.low_t_max, device)
    noise = torch.randn_like(x0)
    if args.identity_frac > 0:
        idm = torch.rand(n, device=device) < args.identity_frac
        noise[idm] = 0.0                       # clean input -> target eps 0
    x_t = schedule.q_sample(x0, t, noise)
    err = (model(x_t, t) - noise) ** 2
    if args.bright_weight > 0:
        w = 1.0 + args.bright_weight * (x0 + 1.0).clamp(min=0) / (net_ceil + 1.0)
        return (w * err).sum() / w.sum()
    return err.mean()


def train(args):
    device = get_device()
    print(f"Device: {device}")

    if args.dataset == "mnist":
        image_size = 32
        dataset = load_mnist(image_size)
        transform_dict = None
    else:
        image_size = 64
        transform = (AsinhTransform.load(args.transform_json)
                     if args.transform_json else None)
        if transform is not None and abs(transform.z_ceil - args.z_ceil) > 1e-9:
            # Old JSONs carry z_ceil=1.05 (flux ceiling 3.3e-2), which CLIPS
            # the brightest stars: 0.0024% of pixels but 28.6% of the flux.
            # The prox then saturates on exactly the sources the data term
            # fights hardest for -> blobbing. Override to the requested
            # ceiling (default 1.45 covers the brightest sharp pixel, 0.955).
            print(f"[transform] overriding z_ceil {transform.z_ceil} -> "
                  f"{args.z_ceil} (b, A, floor unchanged)")
            transform.z_ceil = float(args.z_ceil)
        dataset = StarPatchDataset(
            args.data_dir, blurry_dir=args.blurry_dir, transform=transform,
            image_size=image_size, b_factor=args.b_factor,
            derive_sample=args.derive_sample, max_patches=args.max_patches,
            net_scale=args.net_scale, z_ceil=args.z_ceil,
        )
        transform_dict = dataset.transform.to_dict()
        # persist next to the checkpoint so PnP/eval can load one shared copy
        tpath = os.path.splitext(args.out)[0] + "_transform.json"
        dataset.transform.save(tpath)
        print(f"Saved transform constants to {tpath}")

    # held-out split so a long run can be judged on generalization rather
    # than train loss (and the best checkpoint chosen honestly)
    val_loader = None
    if args.val_frac > 0 and len(dataset) > 100:
        n_val = max(1, int(len(dataset) * args.val_frac))
        n_tr = len(dataset) - n_val
        g = torch.Generator().manual_seed(0)
        train_set, val_set = torch.utils.data.random_split(
            dataset, [n_tr, n_val], generator=g)
        val_loader = DataLoader(val_set, batch_size=args.batch_size,
                                shuffle=False, num_workers=args.num_workers)
        print(f"Split: {n_tr} train / {n_val} val patches")
    else:
        train_set = dataset

    loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        persistent_workers=args.num_workers > 0,
        pin_memory=(device.type == "cuda"),
    )

    dil = tuple(int(v) for v in args.dilations.split(",")) if args.dilations \
        else (1, 2, 3, 4, 4, 3, 2, 1)
    model = FlatCNN(in_channels=1, base=args.base, t_dim=128,
                    dilations=dil).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    rf = 1 + 2 * sum(dil)
    print(f"FlatCNN parameters: {n_params/1e6:.2f}M   "
          f"dilations={dil}  receptive field={rf}px")
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    schedule = Schedule(timesteps=args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # net-space ceiling for the bright-pixel loss ramp (stars only; MNIST
    # lives in [-1, 1] so its ceiling is 1)
    net_ceil = (dataset.transform.net_ceil if transform_dict is not None
                else 1.0)
    print(f"Objective: low_t_frac={args.low_t_frac} (low_t_max={args.low_t_max}), "
          f"identity_frac={args.identity_frac}, "
          f"bright_weight={args.bright_weight}, net_ceil={net_ceil:.2f}")

    if args.overfit_one_batch:
        # Standard first sanity check: loss should approach ~0.
        x0, _ = next(iter(loader))
        x0 = x0.to(device)
        print("Overfitting one batch...")
        for step in range(args.overfit_steps):
            loss = diffusion_loss(model, schedule, x0, args, net_ceil)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == args.overfit_steps - 1:
                print(f"  step {step:5d}  loss {loss.item():.6f}")
        return

    ema = EMA(model, decay=args.ema_decay) if args.ema_decay > 0 else None
    amp_on = args.amp and device.type == "cuda"
    total_steps = args.epochs * max(len(loader), 1)
    print(f"Training: {args.epochs} epochs x {len(loader)} steps = "
          f"{total_steps} steps | EMA={args.ema_decay} | amp={amp_on} | "
          f"warmup={args.warmup_steps} | grad_clip={args.grad_clip}")

    def build_cfg():
        return {
            "base": args.base,
            "timesteps": args.timesteps,
            "image_size": image_size,
            "dataset": args.dataset,
            "transform": transform_dict,
            "dilations": list(dil),
            # PnP-oriented objective settings (for the record)
            "low_t_frac": args.low_t_frac,
            "low_t_max": args.low_t_max,
            "identity_frac": args.identity_frac,
            "bright_weight": args.bright_weight,
        }

    def save(path, extra=None):
        blob = {"model": model.state_dict(), "config": build_cfg()}
        if ema is not None:
            blob["ema"] = ema.state_dict()
        if extra:
            blob.update(extra)
        torch.save(blob, path)

    gstep = 0
    best_val = float("inf")
    best_path = os.path.splitext(args.out)[0] + "_best.pt"
    for epoch in range(args.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for x0, _ in loader:
            x0 = x0.to(device, non_blocking=True)
            if args.channels_last:
                x0 = x0.to(memory_format=torch.channels_last)
            for gp in opt.param_groups:
                gp["lr"] = lr_at(gstep, total_steps, args.lr,
                                 args.warmup_steps, args.lr_schedule)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp_on):
                loss = diffusion_loss(model, schedule, x0, args, net_ceil)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               args.grad_clip)
            opt.step()
            if ema is not None:
                ema.update(model, step=gstep)
            running += loss.item()
            n_batches += 1
            gstep += 1

        msg = (f"epoch {epoch+1:3d}/{args.epochs}  "
               f"train {running/max(n_batches,1):.6f}")

        # ---- validation (uses a FIXED seed so epochs are comparable) ----
        if val_loader is not None and (epoch + 1) % args.val_every == 0:
            model.eval()
            vsum, vn = 0.0, 0
            with torch.no_grad():
                torch.manual_seed(1234)
                for xv, _ in val_loader:
                    xv = xv.to(device, non_blocking=True)
                    vsum += diffusion_loss(model, schedule, xv, args,
                                           net_ceil).item()
                    vn += 1
            vloss = vsum / max(vn, 1)
            msg += f"  val {vloss:.6f}"
            if vloss < best_val:
                best_val = vloss
                save(best_path, {"epoch": epoch + 1, "val_loss": vloss})
                msg += "  <- best"
        print(msg, flush=True)

        save(args.out, {"epoch": epoch + 1})
    print(f"Saved checkpoint to {args.out}")
    if val_loader is not None:
        print(f"Best-by-validation checkpoint: {best_path} "
              f"(val {best_val:.6f})")
    if ema is not None:
        print("Checkpoints contain BOTH 'model' and 'ema' weights; "
              "inference prefers 'ema'.")


# Sampling / visual sanity check
def sample(args):
    device = get_device()
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["config"]
    model = FlatCNN(in_channels=1, base=cfg["base"], t_dim=128).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    schedule = Schedule(timesteps=cfg["timesteps"], device=device)
    n = args.n_samples
    imgs = schedule.p_sample_loop(model, (n, 1, cfg["image_size"], cfg["image_size"]), device)

    tdict = cfg.get("transform")
    if tdict is not None:
        # star model: samples live in MODEL space -- net units when the
        # transform carries the affine, raw z otherwise (legacy checkpoints)
        t = AsinhTransform.from_dict(tdict)
        lo, hi = t.net_floor, t.net_ceil
    else:
        lo, hi = -1.0, 1.0
    imgs = imgs.clamp(lo, hi).cpu().numpy()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = np.atleast_1d(axes).ravel()
    for i in range(len(axes)):
        axes[i].axis("off")
        if i < n:
            axes[i].imshow(imgs[i, 0], cmap="gray", origin="lower",
                           vmin=lo, vmax=hi)
    out_png = os.path.splitext(args.ckpt)[0] + "_samples.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved samples to {out_png}")



def main():
    p = argparse.ArgumentParser(description="Flat CNN diffusion model")
    p.add_argument("--dataset", choices=["mnist", "stars"], required=True)
    p.add_argument("--data-dir", type=str, default=None,
                   help="Directory of SHARP .fits or .npy patches (stars only)")
    p.add_argument("--blurry-dir", type=str, default=None,
                   help="Directory of BLURRY patches; used only to derive the "
                        "asinh range constant A (stars only)")
    p.add_argument("--b-factor", type=float, default=1.5,
                   help="Softening scale b = b_factor * sharp noise sigma")
    p.add_argument("--net-scale", type=float, default=0.20,
                   help="Net-space affine x = z/s - 1: background -> -1, "
                        "data std ~0.46 (raw z std ~0.09 collapses the "
                        "prior to noise). Stored in the transform JSON.")
    p.add_argument("--transform-json", type=str, default=None,
                   help="Load frozen AsinhTransform constants instead of "
                        "deriving them (use for reproducible re-runs)")
    p.add_argument("--derive-sample", type=int, default=4000,
                   help="Number of patches sampled per domain when deriving "
                        "transform constants (memory guard)")
    p.add_argument("--z-ceil", type=float, default=1.45,
                   help="asinh clip ceiling in z. 1.45 -> flux ceiling 1.85, "
                        "covering the brightest sharp pixel (0.955). The old "
                        "1.05 clipped 28.6%% of total flux into saturation, "
                        "making the prox blob the brightest sources.")
    p.add_argument("--low-t-frac", type=float, default=0.5,
                   help="fraction of training draws taken from the LOW-t "
                        "range [0, low-t-max) -- the noise levels the PnP "
                        "prox actually operates at. The rest are uniform "
                        "over all T (wide coverage).")
    p.add_argument("--low-t-max", type=int, default=75,
                   help="upper end of the emphasized low-noise range")
    p.add_argument("--identity-frac", type=float, default=0.15,
                   help="fraction of draws trained with ZERO noise and "
                        "target eps=0: teaches the denoiser to leave clean "
                        "star fields untouched (the PnP fixed-point / "
                        "'distance to manifold' property).")
    p.add_argument("--bright-weight", type=float, default=4.0,
                   help="extra loss weight on bright pixels, ramping 1 -> "
                        "1+w from background to ceiling; counteracts asinh "
                        "peak compression (peak-suppression blobbing). "
                        "0 disables.")
    # -- long-run training machinery -----------------------------------------
    p.add_argument("--ema-decay", type=float, default=0.9999,
                   help="EMA decay for the averaged weights (0 disables). "
                        "Usually the single biggest quality gain on a long "
                        "run; inference prefers these weights.")
    p.add_argument("--warmup-steps", type=int, default=1000,
                   help="linear LR warmup steps before the cosine decay")
    p.add_argument("--lr-schedule", choices=["cosine", "none"],
                   default="cosine")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="gradient-norm clip (0 disables)")
    p.add_argument("--amp", action="store_true",
                   help="bfloat16 autocast on CUDA (faster, lets you go wider)")
    p.add_argument("--channels-last", action="store_true",
                   help="channels_last memory format (faster convs on CUDA)")
    p.add_argument("--val-frac", type=float, default=0.02,
                   help="held-out fraction for validation (0 disables); the "
                        "best-by-val checkpoint is saved as <out>_best.pt")
    p.add_argument("--val-every", type=int, default=1,
                   help="run validation every N epochs")
    p.add_argument("--dilations", type=str, default=None,
                   help="comma-separated dilation pattern, e.g. "
                        "'1,2,3,4,6,8,8,6,4,3,2,1'. More/larger entries = "
                        "deeper net and wider receptive field "
                        "(RF = 1 + 2*sum). Default '1,2,3,4,4,3,2,1' -> 41px.")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--base", type=int, default=64, help="Channel width")
    p.add_argument("--timesteps", type=int, default=300)
    p.add_argument("--num-workers", type=int, default=0,
                   help="DataLoader workers; keep 0 on memory-limited "
                        "machines (workers fork the whole process)")
    p.add_argument("--max-patches", type=int, default=None,
                   help="Train on a random subsample of this many patches "
                        "(local sanity checks; omit for full runs)")
    p.add_argument("--out", type=str, default="flat_cnn_diffusion.pt")
    p.add_argument("--overfit-one-batch", action="store_true",
                   help="Sanity check: overfit a single batch")
    p.add_argument("--overfit-steps", type=int, default=1000)
    p.add_argument("--sample-only", action="store_true")
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--n-samples", type=int, default=16)
    args = p.parse_args()

    if args.dataset == "stars" and not args.sample_only:
        if args.data_dir is None:
            p.error("--data-dir is required for --dataset stars")
        if args.blurry_dir is None and args.transform_json is None:
            p.error("stars training needs --blurry-dir (to derive the asinh "
                    "transform) or --transform-json (to load frozen constants)")
    if args.sample_only:
        if args.ckpt is None:
            p.error("--ckpt is required with --sample-only")
        sample(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
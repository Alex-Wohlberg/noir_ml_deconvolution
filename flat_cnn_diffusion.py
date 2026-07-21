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
                 transform: AsinhTransform):
        self.model = model
        self.schedule = schedule
        self.transform = transform

    @torch.no_grad()
    def __call__(self, x_flux: torch.Tensor, sigma_flux: float) -> torch.Tensor:
        x = self.transform.to_net(x_flux)
        sigma_model = self.transform.sigma_flux_to_model(sigma_flux)
        t = self.schedule.sigma_to_t(sigma_model)
        if t > 0:
            abar = self.schedule.alpha_bars[t]
            noise = torch.randn_like(x)
            x_t = abar.sqrt() * x + (1 - abar).sqrt() * noise  # on-distribution
            x = self.schedule.denoise_x0(self.model, x_t, t)

        # Clamp to the trained range before inverting: sinh explodes beyond
        # it, and the denoiser output is not intrinsically bounded. Values
        # outside [net_floor, net_ceil] are outside the training
        # distribution anyway -- pinning them is the correct prior behavior.
        x = x.clamp(self.transform.net_floor, self.transform.net_ceil)
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
                 net_scale: float = 0.20):
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
                net_scale=net_scale
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
        dataset = StarPatchDataset(
            args.data_dir, blurry_dir=args.blurry_dir, transform=transform,
            image_size=image_size, b_factor=args.b_factor,
            derive_sample=args.derive_sample, max_patches=args.max_patches,
            net_scale=args.net_scale,
        )
        transform_dict = dataset.transform.to_dict()
        # persist next to the checkpoint so PnP/eval can load one shared copy
        tpath = os.path.splitext(args.out)[0] + "_transform.json"
        dataset.transform.save(tpath)
        print(f"Saved transform constants to {tpath}")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )

    model = FlatCNN(in_channels=1, base=args.base, t_dim=128).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"FlatCNN parameters: {n_params/1e6:.2f}M")

    schedule = Schedule(timesteps=args.timesteps, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if args.overfit_one_batch:
        # Standard first sanity check: loss should approach ~0.
        x0, _ = next(iter(loader))
        x0 = x0.to(device)
        print("Overfitting one batch...")
        for step in range(args.overfit_steps):
            t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)
            loss = F.mse_loss(model(x_t, t), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == args.overfit_steps - 1:
                print(f"  step {step:5d}  loss {loss.item():.6f}")
        return

    for epoch in range(args.epochs):
        model.train()
        running, n_batches = 0.0, 0
        for x0, _ in loader:
            x0 = x0.to(device)
            t = torch.randint(0, schedule.timesteps, (x0.shape[0],), device=device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, t, noise)
            loss = F.mse_loss(model(x_t, t), noise)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            n_batches += 1
        print(f"epoch {epoch+1:3d}/{args.epochs}  mean loss {running/n_batches:.6f}")

        torch.save(
            {
                "model": model.state_dict(),
                "config": {
                    "base": args.base,
                    "timesteps": args.timesteps,
                    "image_size": image_size,
                    "dataset": args.dataset,
                    "transform": transform_dict,
                },
            },
            args.out,
        )
    print(f"Saved checkpoint to {args.out}")


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
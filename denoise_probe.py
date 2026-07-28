"""
denoise_probe.py

Isolate the diffusion proximal operator so it can be tested on its own,
outside any deconvolution loop. Same machinery sample_diffusion.py uses to
generate star fields, but instead of sampling from pure noise it applies ONE
(or N) denoising step at a CHOSEN STRENGTH to a GIVEN input image.

The point: if a sharp, point-source image goes in and a blobby image comes
out, the prox is the source of the blobbing -- no deconvolution involved.

    x_flux --to_net--> x --(+noise at t)--> x_t --denoise--> x0_hat --> flux

Every stage is exposed and switchable so you can attribute the blobbing:
  --t / --sigma      prox strength (timestep, or flux-space sigma)
  --steps N          1 = one-step Tweedie (the PnP default), N>1 = DDIM
  --no-noise         SKIP the noise injection, denoise the input directly.
                     Isolates "does the denoiser itself blur?" from "does
                     the injected noise destroy the point sources?"
  --no-clamp         skip the net-range clamp (AsinhProx pins to
                     [net_floor, net_ceil]; ruled in/out as a contributor)
  --seed             the injected noise is random; vary it to see the
                     run-to-run spread at a fixed strength

Reports source CONCENTRATION (central pixel / 7x7 sum, 1.0 = perfect point
source) for input and output, which is the quantitative version of "blobby".

Usage:
  # single strength, one step, on the true sharp patch:
  python denoise_probe.py --input patches/sharp/fits/grid_00002_original.fits --t 50

  # sweep strengths to find where the blobbing sets in:
  python denoise_probe.py --input patches/sharp/fits/grid_00002_original.fits \
      --sweep 5,20,50,100,200,299

  # is it the noise or the network? compare:
  python denoise_probe.py --input ... --t 50 --no-noise
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
# NOTE: deliberately NOT calling matplotlib.use("Agg") at import time --
# importing this module from a notebook/IPython session must not hijack the
# interactive backend. The script sets Agg itself in main().
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm

from pnp_deconvolve import load_model


# ----------------------------------------------------------------------------
# The probe
# ----------------------------------------------------------------------------
class DiffusionDenoiser:
    """Apply the trained diffusion prior as a denoiser to an arbitrary image.

    This is AsinhProx unrolled, with each stage individually switchable.
    """

    def __init__(self, ckpt="checkpoints/flat_cnn_stars_v2.pt",
                 transform_json="checkpoints/flat_cnn_stars_v2_transform.json",
                 device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        (self.model, self.schedule, self.transform,
         self.cfg) = load_model(ckpt, transform_json, self.device)
        self.T = self.cfg["timesteps"]

    # -- strength conversions -------------------------------------------------
    def t_from_sigma(self, sigma_flux):
        """Flux-space sigma -> diffusion timestep."""
        return self.schedule.sigma_to_t(
            self.transform.sigma_flux_to_model(sigma_flux))

    def sigma_from_t(self, t):
        """Diffusion timestep -> flux-space sigma."""
        ab = self.schedule.alpha_bars[int(t)]
        sigma_model = float(((1 - ab) / ab).sqrt())
        return (sigma_model * self.transform.net_scale
                * self.transform.b * self.transform.A)

    def noise_level(self, t):
        """(injected noise std, signal retained) at timestep t."""
        ab = float(self.schedule.alpha_bars[int(t)])
        return (1 - ab) ** 0.5, ab ** 0.5

    # -- the actual operation -------------------------------------------------
    @torch.no_grad()
    def denoise(self, x_flux, t=None, sigma=None, steps=1, add_noise=True,
                clamp=True, seed=None, return_stages=False):
        """Denoise a 2-D flux image at strength t (or sigma).

        add_noise=False denoises the input as-is (no forward diffusion),
        which asks "what does this network do to a clean image?".
        """
        if (t is None) == (sigma is None):
            raise ValueError("give exactly one of t / sigma")
        if t is None:
            t = self.t_from_sigma(sigma)
        t = int(np.clip(t, 0, self.T - 1))
        if seed is not None:
            torch.manual_seed(seed)

        xt_in = torch.from_numpy(np.asarray(x_flux, dtype=np.float32)
                                 .copy())[None, None].to(self.device)
        x_net = self.transform.to_net(xt_in)                    # flux -> net

        if add_noise and t > 0:
            abar = self.schedule.alpha_bars[t]
            noise = torch.randn_like(x_net)
            x_t = abar.sqrt() * x_net + (1 - abar).sqrt() * noise
        else:
            x_t = x_net

        if t > 0:
            if steps > 1:
                out_net = self.schedule.denoise_multistep(
                    self.model, x_t, t, n_steps=steps)
            else:
                out_net = self.schedule.denoise_x0(self.model, x_t, t)
        else:
            out_net = x_t

        if clamp:
            out_net = out_net.clamp(self.transform.net_floor,
                                    self.transform.net_ceil)
        out_flux = self.transform.net_to_flux(out_net)[0, 0].cpu().numpy()

        if return_stages:
            return out_flux, {
                "t": t,
                "noisy_net": x_t[0, 0].cpu().numpy(),
                "noisy_flux": self.transform.net_to_flux(x_t)[0, 0].cpu().numpy(),
            }
        return out_flux


# ----------------------------------------------------------------------------
# Metrics + display
# ----------------------------------------------------------------------------
def concentration(img, thresh_nsigma=5.0, r=3):
    """Median (central px / (2r+1)^2 sum) over detected sources.
    1.0 = perfect single-pixel point source; lower = blobbier."""
    from photutils.detection import find_peaks
    from astropy.stats import sigma_clipped_stats
    _, med, std = sigma_clipped_stats(img, sigma=3.0)
    tbl = find_peaks(img, threshold=med + thresh_nsigma * std, box_size=5)
    if tbl is None or len(tbl) == 0:
        return np.nan, 0
    H, W = img.shape
    vals = []
    for row in tbl:
        x0, y0 = int(row["x_peak"]), int(row["y_peak"])
        if r <= x0 < W - r and r <= y0 < H - r:
            st = img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
            s = st.sum()
            if s > 0:
                vals.append(st[r, r] / s)
    return (float(np.median(vals)) if vals else np.nan), len(vals)


def _sig_std(a, iters=5, k=3.0):
    a = np.asarray(a, np.float64).ravel().copy()
    for _ in range(iters):
        m, s = np.median(a), np.std(a)
        keep = np.abs(a - m) < k * s
        if keep.sum() in (0, a.size):
            break
        a = a[keep]
    return float(np.std(a))


def show(ax, img, title):
    lw = max(_sig_std(img), 1e-30)
    ax.imshow(img, origin="lower", cmap="gray",
              norm=AsinhNorm(linear_width=lw, vmin=float(img.min()),
                             vmax=float(img.max())))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def load_image(path):
    if path.endswith(".npy"):
        a = np.load(path)
    else:
        from astropy.io import fits
        a = fits.getdata(path)
    return np.nan_to_num(np.asarray(a, dtype=np.float64).squeeze())


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Apply the diffusion prox to a given image at a given "
                    "strength (isolated from any deconvolution loop)")
    p.add_argument("--input", required=True, help="image .fits or .npy")
    p.add_argument("--ckpt", default="checkpoints/flat_cnn_stars_v2.pt")
    p.add_argument("--transform-json",
                   default="checkpoints/flat_cnn_stars_v2_transform.json")
    p.add_argument("--t", type=int, default=None,
                   help="prox strength as a diffusion timestep (0..T-1)")
    p.add_argument("--sigma", type=float, default=None,
                   help="prox strength as a flux-space sigma (alt. to --t)")
    p.add_argument("--sweep", default=None,
                   help="comma-separated timesteps, e.g. 5,20,50,100,200,299")
    p.add_argument("--steps", type=int, default=1,
                   help="1 = one-step Tweedie (PnP default); N>1 = DDIM")
    p.add_argument("--no-noise", dest="add_noise", action="store_false",
                   help="skip noise injection; denoise the input directly")
    p.add_argument("--no-clamp", dest="clamp", action="store_false",
                   help="skip the net-range clamp")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="denoise_probe_out")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    x_in = load_image(args.input)
    d = DiffusionDenoiser(args.ckpt, args.transform_json)
    base = os.path.splitext(os.path.basename(args.input))[0]
    c_in, n_in = concentration(x_in)
    print(f"Input {args.input}  shape {x_in.shape}")
    print(f"  flux {x_in.sum():.4e}  max {x_in.max():.4e}")
    print(f"  concentration {c_in:.3f} over {n_in} sources "
          f"(1.0 = single-pixel point source)")
    print(f"Model: T={d.T}, one-step={args.steps == 1}, "
          f"noise injection={args.add_noise}, clamp={args.clamp}\n")

    ts = ([int(v) for v in args.sweep.split(",")] if args.sweep
          else [args.t if args.t is not None else d.t_from_sigma(args.sigma)])

    print(f"{'t':>5}{'sigma_flux':>12}{'noise std':>11}{'keeps':>7}"
          f"{'conc':>7}{'nsrc':>6}{'flux/in':>9}{'max/in':>8}")
    outs = []
    for t in ts:
        out = d.denoise(x_in, t=t, steps=args.steps,
                        add_noise=args.add_noise, clamp=args.clamp,
                        seed=args.seed)
        outs.append((t, out))
        nstd, keep = d.noise_level(t)
        c_out, n_out = concentration(out)
        print(f"{t:5d}{d.sigma_from_t(t):12.3e}{nstd:11.3f}{keep:7.3f}"
              f"{c_out:7.3f}{n_out:6d}"
              f"{out.sum()/max(x_in.sum(),1e-30):9.3f}"
              f"{out.max()/max(x_in.max(),1e-30):8.3f}")

    # save the single-strength result
    if not args.sweep:
        from astropy.io import fits as afits
        t0, o0 = outs[0]
        afits.PrimaryHDU(o0.astype(np.float32)).writeto(
            os.path.join(args.out_dir, f"{base}_denoised_t{t0}.fits"),
            overwrite=True)
        np.save(os.path.join(args.out_dir, f"{base}_denoised_t{t0}.npy"),
                o0.astype(np.float32))

    # figure: input + each output
    n = len(outs)
    fig, ax = plt.subplots(1, n + 1, figsize=(3.6 * (n + 1), 3.9))
    ax = np.atleast_1d(ax)
    show(ax[0], x_in, f"INPUT\nconc={c_in:.3f}")
    for i, (t, out) in enumerate(outs):
        c_out, _ = concentration(out)
        show(ax[i + 1], out, f"t={t} ({'noise+' if args.add_noise else ''}"
                             f"{args.steps}-step)\nconc={c_out:.3f}")
    fig.suptitle(f"diffusion prox applied to {os.path.basename(args.input)} "
                 f"-- concentration 1.0 = point source", fontsize=10)
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"{base}_probe.png")
    fig.savefig(png, dpi=135)
    plt.close(fig)
    print(f"\nWrote {png}")
    if not args.sweep:
        print(f"Wrote {os.path.join(args.out_dir, base)}_denoised_t{ts[0]}.fits")


if __name__ == "__main__":
    main()

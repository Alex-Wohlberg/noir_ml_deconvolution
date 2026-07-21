"""
poisson_pnp_deconvolve.py

P4IP-style deconvolution (Rond, Giryes & Elad 2015, arXiv:1511.02500):
Plug-and-Play ADMM with the POISSON log-likelihood as the data term,
using the same asinh-space diffusion prior as pnp_deconvolve.py.

The ADMM iteration (paper Algorithm 1, deblurring variant):

    x-step:  x^{k+1} = argmin_x  -y^T ln(Cx) + 1^T Cx
                                 + (lam/2) ||x - (v^k - u^k)||^2
             (no closed form when C is a blur; solved by inner projected
              gradient iterations with the paper's eps-guard on ln(Cx),
              their Eq. 18 surrogate idea)
    v-step:  v^{k+1} = GaussianDenoise(x^{k+1} + u^k), sigma^2 = beta/lam
             (here: AsinhProx -- the diffusion prior, unchanged; P4IP's
              whole point is the denoiser is a black box)
    u-step:  u^{k+1} = u^k + x^{k+1} - v^{k+1}
    lam <- lam * lam_step   (paper found increasing lam beats constant)

DELIBERATE EXPERIMENTAL CHOICES (comparison study, per project decision):
  - y is CLIPPED AT ZERO before the solve. The Poisson likelihood requires
    y >= 0; our sky-subtracted data has negative noise pixels. Clipping
    makes the noise asymmetric and biases the faint end upward -- this is
    a KNOWN, accepted distortion for the purpose of comparing against the
    L2 pipeline, not a recommended default.
  - Flux is rescaled to pseudo-counts (y_counts = y_clip * scale, with
    scale set so the brightest pixel ~ --peak). The Poisson model is a
    counts model; its noise level is meaningless in arbitrary flux units.
    Reconstruction is scaled back to flux units on output.

Usage:
  python poisson_pnp_deconvolve.py \
      --ckpt flat_cnn_stars.pt --transform-json flat_cnn_stars_transform.json \
      --input patches/blurry/fits/grid_00002.fits \
      --psf-sigma 2.2 --peak 1000 --outer 20 --out-dir p4ip_out
"""

import argparse
import os

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astro_transforms import AsinhTransform
from flat_cnn_diffusion import FlatCNN, Schedule, AsinhProx, get_device
from pnp_deconvolve import PSFOperator, gaussian_kernel, load_model, show

def sigma_clipped_std(a, iters=5, k=3.0):
    a = np.asarray(a, dtype=np.float64).ravel().copy()
    for _ in range(iters):
        med, std = np.median(a), np.std(a)
        keep = np.abs(a - med) < k * std
        if keep.sum() == 0:
            return float(std)
        a = a[keep]
    return float(np.std(a))

def norm01(a):
        lo, hi = a.min(), a.max()
        return (a - lo) / max(hi - lo, 1e-30)

def poisson_nll(x, y, op, eps):
    """-y^T ln(Cx) + 1^T Cx   (paper Eq. 14, up to the y-only constant)."""
    Cx = op.C(x)
    return (-(y * torch.log(torch.clamp(Cx, min=eps))).sum() + Cx.sum())


def x_step(y, op, target, lam, x_init, n_inner=60, lr=None, eps=1e-8):
    """Inner solve of the Poisson x-step by projected gradient descent.

    minimizes  -y^T ln(Cx) + 1^T Cx + lam/2 ||x - target||^2   s.t. x >= 0

    The clamp inside the log is the practical form of the paper's Eq. 18
    surrogate: it keeps the objective finite as (Cx)_i -> 0 while leaving
    the minimum unchanged for small enough eps.
    """
    x = x_init.clone().clamp(min=0).requires_grad_(True)
    if lr is None:
        # crude but serviceable: gradient magnitudes ~ y/Cx are O(1) after
        # count scaling; the quadratic term has curvature lam
        lr = 1.0 / (1.0 + lam)
    opt = torch.optim.Adam([x], lr=lr)
    for _ in range(n_inner):
        opt.zero_grad()
        f = poisson_nll(x, y, op, eps) + 0.5 * lam * (x - target).pow(2).sum()
        f.backward()
        opt.step()
        with torch.no_grad():
            x.clamp_(min=0)  # Poisson intensity is nonnegative
    return x.detach()


def p4ip(y_counts, op, prox: AsinhProx, counts_scale: float,
         n_outer=20, n_inner=60, beta=1.0, lam0=None, lam_step=1.3,
         eps=1e-8, verbose=True):
    """Run P4IP ADMM. y_counts: (1,1,H,W), nonnegative pseudo-counts.

    counts_scale converts flux -> counts; the diffusion prior operates in
    flux units, so the v-step converts down and back up around AsinhProx.
    """
    device = y_counts.device
    if lam0 is None:
        # sigma0 = sqrt(beta/lam0) ~ sqrt(peak): start the denoiser at
        # roughly the photon-noise level of the brightest pixels
        peak = float(y_counts.max())
        lam0 = beta / max(peak, 1.0)
    lam = lam0

    x = y_counts.clone()          # init at the observation (paper sec 3.1.1)
    v = x.clone()
    u = torch.zeros_like(x)
    history = []

    for k in range(n_outer):
        # ---- x-step: Poisson likelihood + quadratic pull toward v - u ----
        x = x_step(y_counts, op, (v - u), lam, x_init=x,
                   n_inner=n_inner, eps=eps)

        # ---- v-step: black-box Gaussian denoiser = diffusion prior -------
        sigma_counts = float(np.sqrt(beta / lam))
        xu_flux = (x + u) / counts_scale
        v = prox(xu_flux, sigma_flux=sigma_counts / counts_scale) * counts_scale
        v = v.clamp(min=0)

        # ---- u-step ------------------------------------------------------
        u = u + x - v

        nll = float(poisson_nll(x, y_counts, op, eps))
        history.append({"iter": k, "poisson_nll": nll,
                        "sigma_counts": sigma_counts, "lam": lam})
        if verbose:
            print(f"  outer {k+1:3d}/{n_outer}  poisson NLL = {nll:.6e}  "
                  f"sigma = {sigma_counts:.3f} counts  lam = {lam:.3e}")
        lam *= lam_step

    return v, history   # v is the denoised iterate; return it as the estimate


def forward_check(y_flux: np.ndarray, truth_flux: np.ndarray,
                  op: PSFOperator, device, out_png: str):
    """Pair-verification: does PSF * truth actually reproduce y?

    Plots truth | C(truth) | y | residual, and prints flux ratio, relative
    residual, and any registration shift (peak of the cross-correlation).
    If the pairs and the assumed PSF are right, C(truth) and y should agree
    to the noise level, the flux ratio should be ~1, and the shift (0, 0).
    """
    t = torch.from_numpy(truth_flux.astype(np.float32))[None, None].to(device)
    ct = op.C(t)[0, 0].cpu().numpy()

    flux_ratio = float(y_flux.sum() / max(ct.sum(), 1e-30))
    resid = y_flux - ct
    rel_resid = float(np.linalg.norm(resid) / max(np.linalg.norm(ct), 1e-30))

    # registration check via FFT cross-correlation peak
    F1 = np.fft.fft2(y_flux - y_flux.mean())
    F2 = np.fft.fft2(ct - ct.mean())
    xc = np.fft.fftshift(np.fft.ifft2(F1 * np.conj(F2)).real)
    py, px = np.unravel_index(np.argmax(xc), xc.shape)
    shift = (py - y_flux.shape[0] // 2, px - y_flux.shape[1] // 2)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    show(axes[0], truth_flux, "true sharp x")
    show(axes[1], ct, "C(truth) = PSF * sharp")
    show(axes[2], y_flux, "observed blurry y")
    vmax = np.percentile(np.abs(resid), 99.5)
    axes[3].imshow(resid, origin="lower", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
    axes[3].set_title("residual  y - C(truth)", fontsize=9)
    axes[3].axis("off")
    fig.suptitle(f"Forward-model check: flux(y)/flux(C truth) = "
                 f"{flux_ratio:.3f}   rel. residual = {rel_resid:.3f}   "
                 f"shift = {shift}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"[forward check] flux(y) / flux(C @ truth) = {flux_ratio:.4f}  "
          f"(should be ~1; ~21 means the flux bug is still in)")
    print(f"[forward check] relative residual ||y - C(truth)|| / ||C(truth)||"
          f" = {rel_resid:.4f}  (noise-level small if pair + PSF are right)")
    print(f"[forward check] registration shift (dy, dx) = {shift}  "
          f"(nonzero means the pair is misaligned)")
    print(f"[forward check] wrote {out_png}")
    return flux_ratio, rel_resid, shift


def main():
    p = argparse.ArgumentParser(description="P4IP Poisson-likelihood "
                                            "deconvolution with diffusion prior")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--transform-json", required=True)
    p.add_argument("--input", required=True, help="blurry FITS patch")
    p.add_argument("--truth", default=None)
    p.add_argument("--psf-file", default=None)
    p.add_argument("--psf-sigma", type=float, default=None)
    p.add_argument("--peak", type=float, default=1000.0,
                   help="pseudo-count value assigned to the brightest pixel "
                        "of y (sets the Poisson noise regime)")
    p.add_argument("--outer", type=int, default=20)
    p.add_argument("--inner", type=int, default=60)
    p.add_argument("--beta", type=float, default=1.0, help="prior weight")
    p.add_argument("--lam0", type=float, default=None)
    p.add_argument("--lam-step", type=float, default=1.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="p4ip_out")
    args = p.parse_args()

    if (args.psf_file is None) == (args.psf_sigma is None):
        p.error("provide exactly one of --psf-file / --psf-sigma")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = get_device()
    print(f"Device: {device}")

    model, schedule, transform, cfg = load_model(
        args.ckpt, args.transform_json, device)
    prox = AsinhProx(model, schedule, transform)

    from astropy.io import fits as afits
    y_flux = afits.getdata(args.input).astype(np.float32)

    # ---- deliberate clip + count scaling (see module docstring) ----------
    n_neg = int((y_flux < 0).sum())
    y_clip = np.clip(y_flux, 0.0, None)
    counts_scale = args.peak / max(float(y_clip.max()), 1e-30)
    y_counts = torch.from_numpy(y_clip * counts_scale)[None, None].to(device)
    print(f"Clipped {n_neg} negative pixels "
          f"({100*n_neg/y_flux.size:.1f}% of patch) to zero [experimental "
          f"choice]; counts scale = {counts_scale:.4g} (peak {args.peak})")

    if args.psf_file:
        kernel = afits.getdata(args.psf_file).astype(np.float64)
    else:
        kernel = gaussian_kernel(args.psf_sigma)
    op = PSFOperator(kernel, device)

    if args.truth:
        truth_flux = afits.getdata(args.truth).astype(np.float32)
        base = os.path.splitext(os.path.basename(args.input))[0]
        forward_check(y_flux, truth_flux, op, device,
                      os.path.join(args.out_dir, f"{base}_forward_check.png"))

    print(f"Running P4IP: {args.outer} outer x {args.inner} inner iters")
    x_counts, history = p4ip(
        y_counts, op, prox, counts_scale,
        n_outer=args.outer, n_inner=args.inner, beta=args.beta,
        lam0=args.lam0, lam_step=args.lam_step,
    )
    x_np = (x_counts[0, 0].cpu().numpy() / counts_scale).astype(np.float32)

    base = os.path.splitext(os.path.basename(args.input))[0]
    fits_out = os.path.join(args.out_dir, f"{base}_p4ip.fits")
    afits.writeto(fits_out, x_np, overwrite=True)

    n_panels = 3 if args.truth else 2
    fig, axes = plt.subplots(1, n_panels + 1, figsize=(4 * (n_panels + 1), 4))
    show(axes[0], y_flux, "blurry input y")
    show(axes[1], x_np, f"P4IP reconstruction\n({args.outer} outer iters)")
    if args.truth:

        truth_np = afits.getdata(args.truth).astype(np.float32)
        sigma = sigma_clipped_std(truth_np)
        eps = max(sigma, 1e-30)
        show(axes[2], norm01(np.arcsinh(truth_np / (10 * eps))), "true sharp")
    ax = axes[-1]
    ax.semilogy([h["poisson_nll"] - min(h2["poisson_nll"] for h2 in history)
                 + 1e-12 for h in history])
    ax.set_xlabel("outer iteration")
    ax.set_ylabel("Poisson NLL (offset)")
    ax.set_title("data-fidelity objective", fontsize=9)
    fig.tight_layout()
    png_out = os.path.join(args.out_dir, f"{base}_p4ip.png")
    fig.savefig(png_out, dpi=150)
    plt.close(fig)

    print(f"\nWrote {fits_out}")
    print(f"Wrote {png_out}")


if __name__ == "__main__":
    main()
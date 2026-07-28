"""
deconv_diffusion_pnp.py   (rewired from deconv_sparse_03.py)

Same SCICO proximal-gradient deconvolution as deconv_sparse_03.py -- L2 data
fidelity, AcceleratedPGM solver -- but the regularizer's PROXIMAL OPERATOR is
the trained flat-CNN diffusion denoiser (flat_cnn_diffusion.AsinhProx, the v2
model) instead of the hand-crafted non-negative L1 norm. This is Plug-and-Play:
the diffusion model is dropped straight into the `g.prox` slot of the solver.

Prox temperature schedule (the ask):
  * the prox "strength" IS the noise level injected before denoising (one knob);
  * it STARTS at the maximum training temperature -- the timestep T-1 that the
    diffusion model was trained on, i.e. the largest noise the prior ever saw;
  * and it DECREASES geometrically over the iterations down to a gentle floor.

The starting sigma is computed from the schedule/transform, not hardcoded:
  sigma_max = sigma_model(T-1) * net_scale * b * A          (flux units)

Input: a 2-D blurry observation as .fits or .npy (this is y). The forward
operator is a Gaussian PSF (--psf-sigma) or a kernel file, matching the blur.

Usage:
  python deconv_diffusion_pnp.py \
      --input convcheck/grid_00002_original_conv.fits \
      --truth patches/sharp/fits/grid_00002_original.fits \
      --psf-sigma 2.0 --maxiter 300 --out-dir diffusion_pnp_out
"""

import argparse
import os
from denoise_probe import DiffusionDenoiser
import numpy as np
import torch

import jax
# The diffusion prox bridges out to PyTorch and carries an annealing counter,
# so it cannot be JAX-traced. Disable jit globally: SCICO's solver step then
# runs eagerly on concrete arrays, and our prox sees real values. Slower in
# principle, but the torch denoiser dominates runtime at 64x64 anyway.
jax.config.update("jax_disable_jit", True)

import matplotlib
matplotlib.use("Agg")   # default: headless (final PNG only). Switched to an
                        # interactive backend by _enable_interactive() when
                        # --show-every is used.
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm


def _enable_interactive():
    """Switch to a GUI backend so per-iteration previews can be displayed.

    Returns True if an interactive backend was found. Without one (e.g. over
    a plain SSH session) plt.show() is a no-op, so we warn rather than
    silently running blind.
    """
    for backend in ("TkAgg", "QtAgg", "Qt5Agg", "GTK3Agg", "MacOSX"):
        try:
            matplotlib.use(backend, force=True)
            return True
        except Exception:
            continue
    return False

import scico.numpy as snp
from scico import functional, linop, loss
from scico.optimize import AcceleratedPGM, PGM
from scico.optimize.pgm import AdaptiveBBStepSize, LineSearchStepSize
from flat_cnn_diffusion import AsinhProx
from pnp_deconvolve import load_model


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def load_image(path):
    if path.endswith(".npy"):
        a = np.load(path)
    else:
        from astropy.io import fits
        a = fits.getdata(path)
    a = np.nan_to_num(np.asarray(a, dtype=np.float32).squeeze())
    if a.ndim != 2:
        raise ValueError(f"expected 2-D image, got {a.shape}")
    return a


def gaussian_kernel(sigma, truncate=4.0):
    r = max(1, int(truncate * sigma + 0.5))
    ax = np.arange(-r, r + 1)
    xx, yy = np.meshgrid(ax, ax)
    k = np.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
    return (k / k.sum()).astype(np.float32)


# ----------------------------------------------------------------------------
# The diffusion denoiser wrapped as a SCICO regularizer functional.
# Its .prox is the PnP denoiser; the prior value is intractable (has_eval
# False), so the displayed objective is the data term only.
# ----------------------------------------------------------------------------
class DiffusionProx(functional.Functional):
    has_eval = False
    has_prox = True

    def __init__(self, prox_op: DiffusionDenoiser, sigmas: np.ndarray,
                 show_every: int = 0, alpha: int = 1.0):
        super().__init__()
        self._prox_op = prox_op            # AsinhProx (torch, flux space)
        self._sigmas = np.asarray(sigmas, dtype=np.float64)
        self._i = 0                        # annealing step counter
        self.calls = 0
        self.lin = functional.L1Norm()
        # Set to the solver after construction. The schedule MUST be indexed
        # by the solver's true iteration, not by the prox call count: a line
        # search (LineSearchStepSize) evaluates the prox several times per
        # iteration, which otherwise burns through the anneal 2x+ too fast
        # and leaves the whole back half clamped at the final temperature.
        self.solver = None
        # display the reconstruction every N iterations; 0 disables.
        # Each figure BLOCKS until the user closes it.
        self.show_every = int(show_every)
        self._last_shown = -1
        self.alpha = alpha

    def prox(self, v, lam=1.0, **kwargs):
        it = getattr(self.solver, "itnum", None)
        if it is None:
            it = self._i
        sig = float(self._sigmas[min(int(it), len(self._sigmas) - 1)])
        self._i += 1
        self.calls += 1
        v_np = np.asarray(v, dtype=np.float32)       # jax -> numpy (2-D)
        #v_np = np.arcsinh(v_np)
        t_max = max(int(self._prox_op.t_from_sigma(sig)), 1)
        ladder = np.unique(np.geomspace(1, t_max, 10).astype(int))[::-1]  # DESCENDING
        out = self._prox_op.denoise(v_np, t=int(ladder[0]))            # inject once
        # for tt in ladder[1:]:
        #     out = self._prox_op.denoise(out, t=int(tt), add_noise=False)

        if it > int(50):
            out = np.arcsinh(out)
            out = self.lin.prox(v=snp.clip(v_np, 0, None), lam=lam/(1729.816**2))


        #out = self.alpha*v_np + (1.0 - self.alpha)*out
        #out = np.sinh(out)
        #out = self._prox_op.denoise(v_np, sigma=sig, steps=8)   # 2-D in, 2-D out
        #out =  (1 - self.alpha)*out + self.alpha*self.lin.prox(v=snp.clip(v_np, 0, None), lam=lam/1729.816)
        #self.lin.prox(v=snp.clip((self.alpha*(v_np) + (1.0 - self.alpha)*out), 0, None), lam=lam/1729.816)

        if self.show_every and (int(it) != self._last_shown) and \
                (int(it) % self.show_every == 0):
            self._last_shown = int(it)
            self._preview(out, sig, int(it))
        return snp.array(out)

    def _preview(self, x, sig, it):
        """Show the current reconstruction; blocks until the window closes."""
        t = self._prox_op.schedule.sigma_to_t(
            self._prox_op.transform.sigma_flux_to_model(sig))
        fig, ax = plt.subplots(figsize=(6, 6))
        show(ax, x, f"iteration {it}   sigma={sig:.3e}  \n"
                    f"flux={x.sum():.4e}  max={x.max():.4e}"
                    f"\n[close this window to continue]")
        fig.tight_layout()
        plt.show()          # blocking: execution resumes when the user closes it
        plt.close(fig)


# ----------------------------------------------------------------------------
# Temperature schedule anchored at the max training timestep
# ----------------------------------------------------------------------------
def build_sigma_schedule(schedule, transform, T, maxiter, t_end=5,
                         t_start=None, sigma_start=None, sigma_end=None):
    """Geometric flux-space sigma schedule for the prox temperature.

    The prox STRENGTH is the diffusion timestep t: it sets both the noise
    injected before denoising, std sqrt(1 - abar_t), and the level the
    denoiser runs at. Larger t = more noise = more prior, less data.
      t = T-1 (default): the max training temperature; the input is ~98%
          noise, so early iterates are almost entirely prior-driven.
      smaller t_start: keeps more of the current iterate, so the data term
          retains more influence from the outset.

    Specify the endpoints EITHER as timesteps (t_start/t_end) or directly
    as flux-space sigmas (sigma_start/sigma_end); sigmas take precedence.
    """
    ab = schedule.alpha_bars
    def flux_for_t(tt):
        tt = int(np.clip(tt, 0, T - 1))
        sigma_model = float(((1 - ab[tt]) / ab[tt]).sqrt())
        return sigma_model * transform.net_scale * transform.b * transform.A
    if t_start is None:
        t_start = T - 1                      # default: max training temp
    sig_start = float(sigma_start) if sigma_start else flux_for_t(t_start)
    sig_end = float(sigma_end) if sigma_end else flux_for_t(max(1, t_end))
    sigmas = np.geomspace(sig_start, sig_end, maxiter)
    return sigmas, sig_start, sig_end


# ----------------------------------------------------------------------------
# Display
# ----------------------------------------------------------------------------
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
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="SCICO PnP deconvolution with the diffusion prox "
                    "(rewired deconv_sparse_03.py)")
    p.add_argument("--input", required=True, help="blurry observation .fits/.npy")
    p.add_argument("--truth", default=None, help="optional sharp reference")
    p.add_argument("--ckpt", default="checkpoints/flat_cnn_stars_v2.pt")
    p.add_argument("--transform-json",
                   default="checkpoints/flat_cnn_stars_v2_transform.json")
    p.add_argument("--psf-sigma", type=float, default=1.0)
    p.add_argument("--psf-file", default=None)
    p.add_argument("--maxiter", type=int, default=300)
    p.add_argument("--prox-steps", type=int, default=1,
                   help="AsinhProx internal steps (1 = one-step Tweedie)")
    p.add_argument("--t-start", type=int, default=150,
                   help="STARTING prox strength as a diffusion timestep "
                        "(default T-1 = max training temperature, ~98%% noise "
                        "injected). Lower it (e.g. 150, 80) to keep more of "
                        "the current iterate and let the data term lead.")
    p.add_argument("--t-end", type=int, default=5,
                   help="final annealing timestep (temperature floor)")
    p.add_argument("--sigma-start", type=float, default=None,
                   help="starting prox strength directly in FLUX units "
                        "(overrides --t-start)")
    p.add_argument("--sigma-end", type=float, default=None,
                   help="final prox strength in flux units (overrides --t-end)")
    p.add_argument("--no-accel", dest="accel", action="store_false",
                   help="use non-accelerated PGM (steadier with a stochastic "
                        "prox) instead of AcceleratedPGM")
    p.add_argument("--show-every", type=int, default=0,
                   help="display the reconstruction every N iterations; each "
                        "window BLOCKS until you close it (e.g. --show-every 5). "
                        "0 = off (headless, final PNG only).")
    p.add_argument("--out-dir", default="diffusion_pnp_out")
    p.add_argument("--alpha", type=float, default = 1.0)
    args = p.parse_args()

    if args.show_every > 0:
        if _enable_interactive():
            print(f"[preview] showing the reconstruction every "
                  f"{args.show_every} iterations "
                  f"(backend {matplotlib.get_backend()}); "
                  f"close each window to continue")
        else:
            print("[preview] WARNING: no interactive matplotlib backend "
                  "available -- previews cannot be displayed. Install a GUI "
                  "backend (e.g. tkinter/PyQt) or drop --show-every.")
            args.show_every = 0
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, schedule, transform, cfg = load_model(args.ckpt, args.transform_json,
                                                 device)
    T = cfg["timesteps"]
    prox_op = DiffusionDenoiser(ckpt = 'checkpoints/flat_cnn_stars_v2.pt', transform_json="checkpoints/flat_cnn_stars_v2_transform.json", device = device) 
    #prox_op = AsinhProx(model, schedule, transform, n_steps=args.prox_steps)

    y_np = load_image(args.input).astype(np.float32)
    psf = load_image(args.psf_file) if args.psf_file else gaussian_kernel(args.psf_sigma)
    psf = psf / psf.sum()

    # SCICO forward operator + L2 data fidelity (same as the reference)
    y = snp.array(y_np)
    A = linop.Convolve(h=snp.array(psf), input_shape=y_np.shape, mode="same")
    y = A(y_np).astype(np.float32)
    f = loss.SquaredL2Loss(y=y, A=A)

    sigmas, sig_start, sig_end = build_sigma_schedule(
        schedule, transform, T, args.maxiter, t_end=args.t_end,
        t_start=args.t_start, sigma_start=args.sigma_start,
        sigma_end=args.sigma_end)
    g = DiffusionProx(prox_op, sigmas, show_every=args.show_every, alpha = args.alpha)

    print(f"Input {args.input}  shape {y_np.shape}")
    print(f"Prox: diffusion denoiser (v2, n_steps={args.prox_steps})")
    # report the schedule in terms of what the prox actually does: the
    # timestep it maps to and the noise std injected at that timestep
    def describe(sig):
        tt = schedule.sigma_to_t(transform.sigma_flux_to_model(sig))
        ab = float(schedule.alpha_bars[tt])
        return tt, (1 - ab) ** 0.5, ab ** 0.5
    t0, n0, s0 = describe(sig_start)
    t1, n1, s1 = describe(sig_end)
    print(f"Temperature (prox strength = injected-noise level):")
    print(f"  start sigma={sig_start:.4e} -> t={t0}"
          f"{' (MAX training temp)' if t0 >= T - 1 else ''}: "
          f"injects noise std {n0:.3f}, keeps signal {s0:.3f}")
    print(f"  end   sigma={sig_end:.4e} -> t={t1}: "
          f"injects noise std {n1:.3f}, keeps signal {s1:.3f}")
    print(f"  decreasing geometrically over {args.maxiter} iters")
    print(f"Solver: {'AcceleratedPGM' if args.accel else 'PGM'}")

    L0 = linop.operator_norm(A, maxiter=20) ** 2
    Solver = AcceleratedPGM if args.accel else PGM
    solver = Solver(f=f, g=g, L0=float(L0), x0=y,
                    maxiter=args.maxiter, step_size = LineSearchStepSize(),
                    itstat_options={"display": True, "period":50})
    print(f"\nSolving ({args.maxiter} iters)...")
    g.solver = solver          # so the anneal indexes by iteration, not call
    x = np.asarray(solver.solve(), dtype=np.float64)
    hist = solver.itstat_object.history(transpose=True)
    print(f"[check] prox called {g.calls} times over {args.maxiter} iters "
          f"({g.calls / max(args.maxiter, 1):.1f}x per iter; >1 means the "
          f"line search re-evaluates it). Anneal is indexed by iteration, "
          f"so it spans the full run regardless.")

    base = os.path.splitext(os.path.basename(args.input))[0]
    from astropy.io import fits as afits
    afits.PrimaryHDU(x.astype(np.float32)).writeto(
        os.path.join(args.out_dir, f"{base}_diffpnp.fits"), overwrite=True)
    np.save(os.path.join(args.out_dir, f"{base}_diffpnp.npy"), x.astype(np.float32))

    res = float(np.linalg.norm(np.asarray(A(snp.array(x.astype(np.float32)))) - A(y_np))
                / max(np.linalg.norm(y_np), 1e-30))
    print(f"\n[result] ||Cx - y|| / ||y|| = {res:.4f}")
    truth = None
    if args.truth:
        truth = load_image(args.truth).astype(np.float64)
        print(f"[result] flux(x)/flux(truth) = {x.sum()/max(truth.sum(),1e-30):.4f}")
        print(f"[result] RMSE(x, truth)      = {float(np.sqrt(np.mean((x-truth)**2))):.4e}")
        truth = load_image(args.input).astype(np.float64)

    # figure
    npan = 3 if truth is not None else 2
    fig, ax = plt.subplots(1, npan + 1, figsize=(4 * (npan + 1), 4))
    show(ax[0], A(y_np), "blurry input y")
    show(ax[1], x, f"diffusion-PnP deconv\n({args.maxiter} it, anneal t={T-1}->{args.t_end})")
    if truth is not None:
        show(ax[2], truth, "true sharp")
    try:
        ax[-1].semilogy(np.asarray(hist.Residual))
        ax[-1].set_title("PGM residual", fontsize=9)
    except Exception:
        ax[-1].axis("off")
    ax[-1].set_xlabel("iteration")
    fig.tight_layout()
    png = os.path.join(args.out_dir, f"{base}_diffpnp.png")
    fig.savefig(png, dpi=140)
    plt.close(fig)
    print(f"\nWrote {os.path.join(args.out_dir, base + '_diffpnp.fits')}")
    print(f"Wrote {png}")


if __name__ == "__main__":
    main()

"""
admm_diffusion_deconvolve.py

PnP-ADMM deconvolution with a conditional diffusion denoising prior, following
Deutsch, "ADMM-Based Image Deconvolution with Conditioned Diffusion Prior"
(and Venkatakrishnan/Bouman/Wohlberg 2013 for the PnP-ADMM framework itself).

    minimize  0.5||Ax - b||^2 + lam * Psi(z)     s.t.  z - x = 0

ADMM splits this into three cheap steps per outer iteration:

    x^{k+1} = argmin_x 0.5||Ax - b||^2 + (rho/2)||x - (z^k - u^k)||^2
            = F^-1[ (conj(K) B + rho F(z^k - u^k)) / (|K|^2 + rho) ]   EXACT
    z^{k+1} = D_sigma( x^{k+1} + u^k )        <- diffusion prior plugged in here
    u^{k+1} = u^k + x^{k+1} - z^{k+1}         <- dual ascent

How this compares to the RED scheme in red_pnp_deconvolve.py
------------------------------------------------------------
Originally this script existed because RED was losing badly. That gap turned
out to be mostly RED's denoiser, not the RED framework: RED was calling a
single Tweedie step, which supplies almost no sharpening at any timestep.
red_pnp_deconvolve.py now runs a --denoise-steps chain (and needed its lambda
re-tuned 0.03 -> 0.3 as a result), and the two schemes are close. Scored
head-to-head by compare_solvers.py on 16 m31bK50 val patches:

                    conc  int_rmse  edge_rmse  floor%   flux   data_res
    RED (1-step)   0.621    0.6699     3.0258   12.85  0.944    0.1482
    RED (chain)    0.770    0.0753     0.1122    7.49  0.968    0.0478
    ADMM           0.717    0.0388     0.2012    0.06  1.021    0.0761
    truth          0.749    0.0000     0.0000    7.28  1.000    0.0589

ADMM still wins where its structure earns it: interior RMSE (0.0388 vs 0.0753,
driven by source photometry -- RMSE at source pixels is 0.085 vs 0.173) and it
does not overfit the noise (RED's data residual 0.0478 sits BELOW the 0.0589
noise floor, ADMM's 0.0761 above it). Its weaknesses are a +2% flux bias and a
floor fraction of 0.06% against the truth's 7.28%, i.e. it holds the background
off zero where the truth sits on it.

The structural reason ADMM's x-update matters is unchanged: gradient descent
on 0.5||Ax-b||^2 with a band-limited A amplifies exactly the frequencies the
PSF suppresses (with lam=0, 43% of pixels clamped and background scatter 540x
the truth). ADMM's x-update is a single exact Fourier solve with a rho ridge,
so the (|K|^2 + rho) denominator is bounded away from zero and those modes are
damped rather than amplified. What changed is that a strong enough prior can
also hold that instability down, which is what RED at lam=0.3 is doing.

Timestep selection follows the paper's heuristic: pick the t whose noise
variance is closest to the noise variance of the input to the denoiser. In the
DDPM parameterization x_t = sqrt(abar_t) x_0 + sqrt(1-abar_t) eps, dividing
through by sqrt(abar_t) gives an equivalent additive noise std of
    sigma(t) = sqrt((1 - abar_t) / abar_t)
so t is chosen by matching sigma(t) to the ADMM denoising level sqrt(lam/rho).

Domains
-------
x, z, u all live in PHYSICAL FLUX, because A is linear only there. The denoiser
wraps norm.json's forward/inverse around the model call, and clamps the x0
estimate to [0,1] in normalized units before inverting -- the asinh inverse is
explosive (normalized 1.2 -> 14x flux error, 2.0 -> inf in float32).

Usage:
python admm_diffusion_deconvolve.py \
    --ckpt /home/alex/noir_ml/global/ml-decon/checkpoints_cond_diffusion_npy/best.pt \
    --data-dir /home/alex/noir_ml/global/ml-decon/data/m31bK50 \
    --split val --indices 0:16 --psf-file psf_50_true.fits \
    --iters 200 --denoise-steps 4 --rho 3.0 --rho-scale 1.0 --lam 0.01 \
    --eta 0.0 --renoise --t-min 1 --t-max 75 \
    --n-show 0 --log-every 50 --out-dir admm_fft_best
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from red_pnp_deconvolve import (TorchNorm, load_checkpoint, load_kernel,
                                gaussian_kernel, make_otf, parse_indices, show,
                                tweedie_chain)
from cond_sample_npy import find_peaks, concentration
from train_conditional_diffusion import cosine_alpha_bar


# ----------------------------------------------------------------------------
# The paper's timestep heuristic
# ----------------------------------------------------------------------------
def sigma_of_t(alpha_bar: torch.Tensor) -> torch.Tensor:
    """Equivalent additive-noise std of each timestep: sqrt((1-abar)/abar)."""
    return ((1.0 - alpha_bar) / alpha_bar).sqrt()


def t_for_sigma(sigmas: torch.Tensor, sigma: float, t_max: int,
                t_min: int = 1) -> int:
    """t whose noise variance is closest to sigma^2, clamped to [t_min, t_max].

    t_max matters: the checkpoint's identity/fixed-point term was trained with
    identity_t_max=100 and the operator degrades sharply past it (||D(x)-x||
    on true patches is 0.015 at t=50, 0.216 at t=100, 46.6 at t=200).

    t_min matters for a different reason. With --rho-scale > 1, rho grows every
    outer iteration, so sigma = sqrt(lam/rho) shrinks and the matched t walks
    down (29 -> 17 -> 9 -> 4 -> 1 in the reference run). But a chain started at
    t=1 is a near-identity operator -- the denoiser stops contributing exactly
    during the late iterations that are supposed to be refining the solution.
    Flooring t keeps the prior alive to the end.
    """
    t = int(torch.argmin((sigmas - float(sigma)).abs()).item())
    return max(int(t_min), min(t, int(t_max)))


# ----------------------------------------------------------------------------
# z-update: a short conditional reverse chain (the paper's "set number of
# denoising steps"), NOT a single Tweedie step
# ----------------------------------------------------------------------------
@torch.no_grad()
def diffusion_denoise(model, v_flux, y_z, sigma_z, n_steps, alpha_bar, sigmas,
                      ideal_norm, t_max=75, t_min=1, guidance=1.0,
                      has_null=False, eta=0.0, generator=None,
                      clamp=(0.0, 1.0), renoise=False, trace=None):
    """Treat v as a noisy image at normalized-domain noise level sigma_z, map
    that to a timestep, and run the reverse chain from there down to ~0.

    v already CARRIES noise of std sigma_z, so tweedie_chain scales it into the
    diffusion parameterization rather than injecting fresh noise:
        x_t = sqrt(abar_t) * (x_0 + sigma(t) eps) = sqrt(abar_t) * z
    """
    z = ideal_norm.forward(v_flux)
    t0 = t_for_sigma(sigmas, sigma_z, t_max, t_min)
    z0 = tweedie_chain(model, z, y_z, t0, n_steps, alpha_bar,
                       guidance=guidance, has_null=has_null, eta=eta,
                       generator=generator, clamp=clamp, renoise=renoise,
                       trace=trace)
    return ideal_norm.inverse(z0), t0


# ----------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", default="/home/alex/noir_ml/global/ml-decon/"
                                     "checkpoints_cond_diffusion_npy/best.pt")
    p.add_argument("--data-dir", type=Path,
                   default=Path("/home/alex/noir_ml/global/ml-decon/data/m31bK50"))
    p.add_argument("--split", choices=["val", "train"], default="val")
    p.add_argument("--indices", default="0:16")
    p.add_argument("--psf-file", default="psf_50_true.fits")
    p.add_argument("--psf-sigma", type=float, default=None)

    p.add_argument("--iters", type=int, default=30,
                   help="ADMM outer iterations. Far fewer than RED needs "
                        "because the x-update is exact, but each one costs "
                        "--denoise-steps model calls.")
    p.add_argument("--denoise-steps", type=int, default=8,
                   help="reverse-chain steps in the z-update. The paper's "
                        "'set number of denoising steps' per outer iteration. "
                        "1 reproduces the one-step RED behaviour (and its "
                        "0.24 sharpening ceiling); 8-20 is the useful range.")
    p.add_argument("--rho", type=float, default=1.0,
                   help="ADMM penalty. Large rho = trust the prior/split, "
                        "small rho = trust the data solve.")
    p.add_argument("--rho-scale", type=float, default=1.0,
                   help="multiply rho by this each outer iteration "
                        "(continuation; >1 tightens the split over the run)")
    p.add_argument("--lam", type=float, default=0.01,
                   help="prior weight. Enters ONLY through the denoising "
                        "level sigma = sqrt(lam/rho), i.e. the paper's "
                        "noise-variance-matching heuristic.")
    p.add_argument("--t-max", type=int, default=75,
                   help="cap on the matched timestep; keep <= the "
                        "checkpoint's identity_t_max")
    p.add_argument("--t-min", type=int, default=1,
                   help="FLOOR on the matched timestep. With --rho-scale > 1 "
                        "the matched t decays to 1, where the chain is a "
                        "near-identity and the prior stops acting. Raising "
                        "this keeps the denoiser contributing to the end.")
    p.add_argument("--eta", type=float, default=0.0,
                   help="DDIM stochasticity, MUST be in [0, 1]. 0 = fully "
                        "deterministic (measured best for photometry: flux "
                        "0.995 vs 0.972 at eta=1), 1 = full ancestral. Values "
                        "> 1 are NOT 'more exploration' -- they make "
                        "1 - abar_next - s^2 negative, which clamps the "
                        "deterministic direction term to zero and deletes "
                        "every refinement step in the chain.")
    p.add_argument("--avg-last", type=int, default=0,
                   help="average the final K z-iterates into the output "
                        "(Polyak averaging). The z-update is a STOCHASTIC, "
                        "non-contractive operator, so the iteration has no "
                        "fixed point -- it samples a neighbourhood. Measured "
                        "relative oscillation of the primal residual is ~0.5 "
                        "with --renoise. Averaging is the standard variance "
                        "reduction for that and costs no extra model calls. "
                        "0 = off (use the final iterate). Only the OUTPUT is "
                        "averaged; the ADMM recursion itself is untouched.")
    p.add_argument("--renoise", action="store_true",
                   help="do a PROPER forward diffusion at the chain entry, "
                        "x_t = sqrt(abar)*z + sqrt(1-abar)*eps, instead of the "
                        "deterministic rescale. Late in an ADMM run z is much "
                        "cleaner than sigma(t0), so the model sees a too-clean "
                        "input, does nothing, and preserves blur. This is the "
                        "calibrated way to get the regeneration that eta > 1 "
                        "produces by accident -- strength is set by t0 (see "
                        "--t-min), not by an unbounded multiplier.")
    p.add_argument("--eta-end", type=float, default=None,
                   help="anneal eta linearly from --eta to this over the outer "
                        "iterations (also must be in [0,1]). Default: constant.")
    p.add_argument("--guidance", type=float, default=1.0)
    p.add_argument("--nonneg", action="store_true", default=False)
    p.add_argument("--weights", choices=["ema", "raw"], default="ema")
    p.add_argument("--no-fits", dest="write_fits", action="store_false",
                   help="skip writing <out-dir>/fits/{recon,truth}/*.fits "
                        "(they are what the external photometry tools consume)")
    p.add_argument("--n-show", type=int, default=6,
                   help="rows in patches.png, one per patch. 0 or negative = "
                        "ALL the patches in --indices.")
    p.add_argument("--crop", type=int, default=4)
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=Path, default=Path("admm_diff_out"))
    p.add_argument("--trace-chain", type=int, default=0, metavar="N",
                   help="record per-chain-step diagnostics for the first N "
                        "outer iterations and every --log-every-th one after, "
                        "into <out-dir>/chain_trace.csv. The column that "
                        "matters is dz0_rms: how far the chain's x0 estimate "
                        "moves at each step, in normalized units. If it decays "
                        "to ~0 over the chain, the late steps are dead and "
                        "entry-only --renoise cannot reach them (which is what "
                        "eta > 1 was compensating for). 1 normalized unit is "
                        "~14.5 mag at the bright end, so dz0_rms=0.001 is "
                        "~0.015 mag of movement.")
    args = p.parse_args()

    # eta outside [0,1] over-noises relative to the DDIM derivation. NOTE the
    # old claim here -- that it "deletes every refinement step" -- was measured
    # FALSE on 2026-08-11: at t0=75/n_steps=20 the deterministic term is zeroed
    # in 0/19 steps at eta=1.0, 2/19 at eta=1.5, 6/19 at eta=2.0, and only at
    # the tail of the chain where the term is already ~0.006. eta=1.5 injects
    # ~5.6x the total noise of --renoise alone with 17 intact steps. Warn, but
    # do not mis-describe it. See tweedie_chain's docstring.
    for nm, v in (("--eta", args.eta), ("--eta-end", args.eta_end)):
        if v is not None and not (0.0 <= v <= 1.0):
            print(f"warning: {nm}={v} is outside [0, 1]. DDIM's eta is a "
                  f"fraction of the ancestral noise, not a gain, so this "
                  f"over-noises every step and zeroes the deterministic term "
                  f"in the last few. If it helps, that is evidence the chain "
                  f"wants more noise than t_max allows (sigma=sqrt(lam/rho) "
                  f"asks for t=188 at lam=0.01/rho=0.1 and is clamped to "
                  f"t_max); prefer raising --t-max or --renoise, and use "
                  f"--trace-chain to see which steps are actually moving.")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Dump the exact config next to the outputs. Without this a directory of
    # FITS is unattributable after the fact, which has already cost one
    # unreproducible tuning result.
    (args.out_dir / "config.json").write_text(json.dumps(
        {k: (str(v) if isinstance(v, Path) else v)
         for k, v in vars(args).items()}, indent=2, sort_keys=True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    gen = torch.Generator(device=device).manual_seed(args.seed)

    model, T, ck = load_checkpoint(args.ckpt, device, weights=args.weights)
    alpha_bar = cosine_alpha_bar(T).to(device)
    sigmas = sigma_of_t(alpha_bar)
    norms = ck["dataset_norm"] or json.loads(
        (args.data_dir / "norm.json").read_text())
    obs_norm, ideal_norm = TorchNorm(norms["observed"]), TorchNorm(norms["ideal"])
    has_null = float(ck.get("p_uncond") or 0.0) > 0.0

    observed = np.load(args.data_dir / f"{args.split}_observed.npy")
    ideal = np.load(args.data_dir / f"{args.split}_ideal.npy")
    idx = parse_indices(args.indices, len(observed))
    y_z = torch.from_numpy(observed[idx]).float().to(device)
    b = obs_norm.inverse(y_z)                       # measurement, flux units
    truth = ideal_norm.inverse(
        torch.from_numpy(ideal[idx]).float().to(device))
    B, _, H, W = y_z.shape

    kernel = (gaussian_kernel(args.psf_sigma) if args.psf_sigma is not None
              else load_kernel(args.psf_file))
    K = make_otf(kernel, (H, W), device)
    K2 = (K.conj() * K).real
    Kb = K.conj() * torch.fft.rfft2(b)              # conj(K) B, precomputed

    print(f"[data] {args.split}: {len(idx)} patches")
    print(f"[psf]  {'gaussian %.2f' % args.psf_sigma if args.psf_sigma else args.psf_file}")
    print(f"[admm] iters={args.iters} denoise_steps={args.denoise_steps} "
          f"rho={args.rho} lam={args.lam} t_max={args.t_max}")

    # ADMM state, all in flux units
    x = b.clone()
    z = b.clone()
    u = torch.zeros_like(b)
    rho = float(args.rho)
    bn = float(torch.linalg.vector_norm(b))
    truth_np = truth.cpu().numpy()[:, 0]
    peaks = find_peaks(truth_np)
    hist = {"data": [], "prim": [], "dual": [], "conc": []}
    z_acc, n_acc = None, 0
    chain_trace = []

    for k in range(args.iters):
        # ---- x-update: EXACT Fourier solve (this is the whole point) ----
        rhs = Kb + rho * torch.fft.rfft2(z - u)
        x = torch.fft.irfft2(rhs / (K2 + rho), s=(H, W))

        # ---- z-update: short conditional reverse chain ----
        sigma_z = float(np.sqrt(max(args.lam, 1e-12) / max(rho, 1e-12)))
        z_prev = z
        # linear anneal eta -> eta_end over the run. Values above 1 over-noise
        # every step (see the --eta help and tweedie_chain's docstring); they
        # do NOT delete the chain's deterministic path except at its tail.
        eta_k = (args.eta if args.eta_end is None
                 else args.eta + (args.eta_end - args.eta) * k / max(args.iters - 1, 1))
        # Trace the first N iterations densely, then sample -- the interesting
        # contrast is early (iterate still blurry) vs late (iterate converged
        # and, without --renoise, far cleaner than sigma(t0) implies).
        tr = ([] if args.trace_chain and
              (k < args.trace_chain or k % max(args.log_every, 1) == 0)
              else None)
        z, t_used = diffusion_denoise(
            model, x + u, y_z, sigma_z, args.denoise_steps, alpha_bar, sigmas,
            ideal_norm, t_max=args.t_max, t_min=args.t_min,
            guidance=args.guidance, has_null=has_null, eta=eta_k,
            generator=gen, renoise=args.renoise, trace=tr)
        if tr is not None:
            for rec in tr:
                chain_trace.append({"iter": k, "eta": eta_k,
                                    "sigma_z": sigma_z, **rec})
        if args.nonneg:
            z = z.clamp(min=0.0)

        # Polyak averaging of the OUTPUT only -- the recursion below still uses
        # the instantaneous z, so the dynamics are unchanged.
        if args.avg_last > 0 and k >= args.iters - args.avg_last:
            z_acc = z.clone() if z_acc is None else z_acc + z
            n_acc += 1

        # ---- u-update ----
        u = u + x - z

        data_r = float(torch.linalg.vector_norm(
            torch.fft.irfft2(torch.fft.rfft2(z) * K, s=(H, W)) - b)) / max(bn, 1e-30)
        prim = float(torch.linalg.vector_norm(x - z))
        dual = float(rho * torch.linalg.vector_norm(z - z_prev))
        c = concentration(z.cpu().numpy()[:, 0], peaks)
        hist["data"].append(data_r); hist["prim"].append(prim)
        hist["dual"].append(dual); hist["conc"].append(c)
        if k % args.log_every == 0 or k == args.iters - 1:
            print(f"  it {k:3d}  t={t_used:3d} sigma={sigma_z:.4f}  "
                  f"||Az-b||/||b||={data_r:.4f}  prim={prim:.3e} "
                  f"dual={dual:.3e}  conc={c:.3f}  "
                  f"flux/truth={float(z.sum()/truth.sum()):.3f}", flush=True)
        rho *= args.rho_scale

    if n_acc:
        z = z_acc / n_acc
        print(f"[avg] output = mean of the final {n_acc} z-iterates")

    # ---- metrics ----
    c0 = args.crop
    crop = np.s_[c0:H - c0, c0:W - c0]
    z_np = z.cpu().numpy()
    b_np = b.cpu().numpy()
    rows = []
    for i2, kk in enumerate(idx):
        xf, tf = z_np[i2, 0][crop], truth_np[i2][crop]
        rows.append({"index": kk,
                     "flux_ratio": float(xf.sum() / max(tf.sum(), 1e-30)),
                     "peak_ratio": float(xf.max() / max(tf.max(), 1e-30)),
                     "rmse": float(np.sqrt(np.mean((xf - tf) ** 2)))})
    with open(args.out_dir / "metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader()
        w.writerows(rows)
    np.save(args.out_dir / f"{args.split}_recon.npy", z_np.astype(np.float32))
    # convergence history, so the residual behaviour can be analysed instead of
    # only eyeballed in convergence.png
    np.savez(args.out_dir / "history.npz",
             **{k: np.asarray(v, np.float64) for k, v in hist.items()})

    if chain_trace:
        with open(args.out_dir / "chain_trace.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(chain_trace[0]))
            w.writeheader(); w.writerows(chain_trace)
        # Per-step summary averaged over the traced outer iterations: this is
        # the H1-vs-H2 read. A dz0_rms profile that decays to ~0 by the end of
        # the chain means the late steps do nothing and the fix is per-step
        # noise (eta) rather than more entry noise (t_max).
        last_it = max(r["iter"] for r in chain_trace)
        tail = [r for r in chain_trace if r["iter"] == last_it]
        print(f"\n[chain] per-step movement at outer iter {last_it} "
              f"(eta={tail[0]['eta']:.2f}, t0={tail[0]['t']}):")
        print("   step    t   dz0_rms   ~mag   det_coef  noise_coef  clamp_hi%")
        for r in tail:
            dz = r["dz0_rms"]
            print(f"  {r['step']:5d} {r['t']:4d}  {dz:8.5f} "
                  f"{14.475 * dz if dz == dz else float('nan'):6.3f} "
                  f"  {r['det_coef']:8.4f}  {r['noise_coef']:9.4f}  "
                  f"{100 * r['clamp_frac_hi']:8.4f}")
        print(f"[chain] full trace -> {args.out_dir / 'chain_trace.csv'}")

    # ---- FITS, for the external photometry tools ----
    # completeness_analysis.py / final_photometry_table.py glob FITS and pair a
    # reconstruction with its truth BY FILENAME, so both trees use identical
    # basenames. Truth is written alongside so this directory is self-contained
    # and can be pointed at directly:
    #     python final_photometry_table.py --fits-dir <out-dir>/fits \
    #            --methods recon
    if args.write_fits:
        from astropy.io import fits as _fits
        for sub, arr in (("recon", z_np[:, 0]), ("truth", truth_np)):
            d = args.out_dir / "fits" / sub
            d.mkdir(parents=True, exist_ok=True)
            # DELETE stale patches first. Without this, re-running with fewer
            # --indices leaves the previous run's extra patch_NN.fits in place
            # and the photometry tools, which just glob the directory, silently
            # pool two different configurations into one score.
            stale = sorted(d.glob("patch_*.fits"))
            for f in stale:
                f.unlink()
            if stale and len(stale) != arr.shape[0]:
                print(f"[fits] cleared {len(stale)} stale patch files in {d} "
                      f"(previous run had a different --indices)")
            for i in range(arr.shape[0]):
                _fits.writeto(d / f"patch_{i:02d}.fits",
                              arr[i].astype(np.float32), overwrite=True)
        print(f"[fits] wrote {args.out_dir}/fits/{{recon,truth}}/patch_*.fits "
              f"({B} patches each)")

    inner = np.s_[:, 16:H - 16, 16:W - 16]
    interior_rmse = float(np.sqrt(
        ((z_np[:, 0][inner] - truth_np[inner]) ** 2).mean()))
    print(f"\n[result] flux_ratio={np.mean([r['flux_ratio'] for r in rows]):.4f}"
          f"  rmse={np.mean([r['rmse'] for r in rows]):.4e}")
    print(f"[result] concentration={concentration(z_np[:, 0], peaks):.3f}"
          f"   (truth {concentration(truth_np, peaks):.3f})")
    print(f"[result] interior rmse={interior_rmse:.4f}")
    print(f"[result] pixels at the zero floor: {100*(z_np <= 0).mean():.2f}%   "
          f"(truth {100*(truth_np <= 0).mean():.2f}% -- MATCH this, do not "
          f"minimize it; see solver_metrics.floor_frac)")
    print("[note] for a like-for-like comparison against the sampler and RED, "
          "run compare_solvers.py -- it scores every method with one metric "
          "definition. The numbers printed here use this script's own peak "
          "list and crop and are NOT comparable to RED's own printout.")

    nshow = B if args.n_show <= 0 else min(B, args.n_show)
    fig, ax = plt.subplots(nshow, 3, figsize=(11, 3.6 * nshow), squeeze=False)
    for i2 in range(nshow):
        show(ax[i2, 0], b_np[i2, 0], f"observed (idx {idx[i2]})")
        show(ax[i2, 1], z_np[i2, 0],
             f"ADMM+diffusion  flux/truth={rows[i2]['flux_ratio']:.3f}")
        show(ax[i2, 2], truth_np[i2], "ideal (truth)")
    fig.tight_layout(); fig.savefig(args.out_dir / "patches.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].semilogy(hist["data"]); ax[0].set_title("||Az-b||/||b||", fontsize=9)
    ax[1].semilogy(hist["prim"], label="primal ||x-z||")
    ax[1].semilogy(hist["dual"], label="dual"); ax[1].legend(fontsize=8)
    ax[1].set_title("ADMM residuals", fontsize=9)
    ax[2].plot(hist["conc"])
    ax[2].axhline(concentration(truth_np, peaks), color="g", ls="--", lw=0.9,
                  label="truth")
    ax[2].legend(fontsize=8); ax[2].set_title("point-source concentration",
                                              fontsize=9)
    for a in ax:
        a.set_xlabel("ADMM iteration"); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(args.out_dir / "convergence.png", dpi=130)
    plt.close(fig)
    print(f"Wrote {args.out_dir}/patches.png ({nshow} of {B} patches), "
          f"convergence.png, metrics.csv")


if __name__ == "__main__":
    main()

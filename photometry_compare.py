"""
photometry_compare.py

Judge a deconvolution by the science it is for -- star POSITIONS and FLUXES --
instead of by eye. Detects sources in the reconstruction and in the sharp
truth, matches them, and reports completeness, false positives, positional
accuracy, and aperture-flux accuracy.

Rationale: the truth patches render stars as ~single bright pixels, so a
reconstruction that puts a star's flux in a 3-px blob "looks wrong" but may be
photometrically perfect -- the centroid and the aperture-summed flux are what a
photometry pipeline actually consumes. This script measures exactly those.

Method
------
* Detection: photutils.find_peaks on each image (robust for both single-pixel
  truth sources and few-pixel reconstruction blobs), threshold =
  median + N*sigma_clipped_std. The TRUTH detections are the reference catalog.
* Sub-pixel position: centroid_com in a small window around each peak.
* Flux: circular-aperture sum (r = --aper px) with local annulus background
  subtraction, on both images at their OWN detected positions and, for matched
  pairs, at the TRUTH position on the reconstruction (forced photometry) so the
  flux comparison is not confounded by centroid disagreement.
* Matching: greedy nearest-neighbour within --match-radius px.

Outputs: printed metrics, a CSV of matched pairs, and an overlay+scatter PNG.

Usage:
  python photometry_compare.py \
      --recon pnp_out_v2/grid_00002_original_pnp.fits \
      --truth patches/sharp/fits/grid_00002_original.fits \
      --out-dir photometry_out
"""

import argparse
import csv
import os

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import find_peaks
from photutils.centroids import centroid_com
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import AsinhNorm


def detect(img, nsigma, box, edge):
    """Return an (N,2) array of (x,y) source positions with sub-pixel centroids."""
    mean, median, std = sigma_clipped_stats(img, sigma=3.0)
    tbl = find_peaks(img, threshold=median + nsigma * std, box_size=box)
    if tbl is None or len(tbl) == 0:
        return np.empty((0, 2)), std
    pos = []
    h, w = img.shape
    r = box // 2
    for row in tbl:
        x0, y0 = int(row["x_peak"]), int(row["y_peak"])
        if x0 < edge or x0 >= w - edge or y0 < edge or y0 >= h - edge:
            continue
        sub = img[y0 - r:y0 + r + 1, x0 - r:x0 + r + 1]
        sub = np.clip(sub - median, 0, None)
        if sub.sum() <= 0:
            pos.append((x0, y0)); continue
        cx, cy = centroid_com(sub)
        pos.append((x0 - r + cx, y0 - r + cy))
    return np.asarray(pos), std


def aper_flux(img, xy, r, r_in, r_out):
    """Background-subtracted circular-aperture flux at each (x,y)."""
    if len(xy) == 0:
        return np.array([])
    ap = CircularAperture(xy, r=r)
    an = CircularAnnulus(xy, r_in=r_in, r_out=r_out)
    phot = aperture_photometry(img, ap)
    # robust local background per source from the annulus
    bkg = []
    an_masks = an.to_mask(method="center")
    for m in an_masks:
        vals = m.multiply(img)[m.data > 0]
        bkg.append(np.median(vals) if len(vals) else 0.0)
    bkg = np.asarray(bkg)
    return np.asarray(phot["aperture_sum"]) - bkg * ap.area


def match(truth_xy, recon_xy, radius):
    """Greedy nearest-neighbour matching. Returns list of (ti, ri, dist)."""
    pairs, used = [], set()
    for ti, t in enumerate(truth_xy):
        if len(recon_xy) == 0:
            break
        d = np.hypot(recon_xy[:, 0] - t[0], recon_xy[:, 1] - t[1])
        for ri in np.argsort(d):
            if ri in used:
                continue
            if d[ri] <= radius:
                pairs.append((ti, ri, float(d[ri]))); used.add(ri)
            break
    return pairs


def main():
    p = argparse.ArgumentParser(description="Photometry comparison of a "
                                            "reconstruction against sharp truth")
    p.add_argument("--recon", required=True)
    p.add_argument("--truth", required=True)
    p.add_argument("--nsigma", type=float, default=5.0,
                   help="detection threshold in sigma above background")
    p.add_argument("--box", type=int, default=5, help="peak-finder box size (px)")
    p.add_argument("--aper", type=float, default=3.0, help="aperture radius (px)")
    p.add_argument("--ann-in", type=float, default=5.0)
    p.add_argument("--ann-out", type=float, default=8.0)
    p.add_argument("--match-radius", type=float, default=2.5,
                   help="max centroid distance to call a match (px)")
    p.add_argument("--edge", type=int, default=4,
                   help="ignore sources within this many px of the border")
    p.add_argument("--out-dir", default="photometry_out")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    recon = fits.getdata(args.recon).astype(np.float64)
    truth = fits.getdata(args.truth).astype(np.float64)

    t_xy, t_std = detect(truth, args.nsigma, args.box, args.edge)
    r_xy, r_std = detect(recon, args.nsigma, args.box, args.edge)
    pairs = match(t_xy, r_xy, args.match_radius)

    # forced photometry on the reconstruction at TRUTH positions (matched only),
    # so flux accuracy isn't confounded by small centroid disagreements
    t_flux_all = aper_flux(truth, t_xy, args.aper, args.ann_in, args.ann_out)
    matched_t_xy = np.array([t_xy[ti] for ti, _, _ in pairs]) if pairs else np.empty((0, 2))
    tf = aper_flux(truth, matched_t_xy, args.aper, args.ann_in, args.ann_out)
    rf = aper_flux(recon, matched_t_xy, args.aper, args.ann_in, args.ann_out)

    n_t, n_r, n_m = len(t_xy), len(r_xy), len(pairs)
    completeness = n_m / n_t if n_t else 0.0
    spurious = (n_r - n_m) / n_r if n_r else 0.0
    dists = np.array([d for _, _, d in pairs])
    with np.errstate(divide="ignore", invalid="ignore"):
        flux_ratio = rf / tf

    print(f"\n=== Photometry: {os.path.basename(args.recon)} vs truth ===")
    print(f"truth sources: {n_t}   recon sources: {n_r}   matched: {n_m}")
    print(f"completeness (matched/truth)  = {completeness:6.1%}")
    print(f"spurious    (unmatched/recon) = {spurious:6.1%}")
    if n_m:
        print(f"position error  median = {np.median(dists):.3f} px   "
              f"90th pct = {np.percentile(dists,90):.3f} px")
        good = np.isfinite(flux_ratio) & (tf > 0)
        fr = flux_ratio[good]
        print(f"aperture flux ratio (recon/truth)  median = {np.median(fr):.3f}   "
              f"scatter (16-84 pct) = [{np.percentile(fr,16):.3f}, "
              f"{np.percentile(fr,84):.3f}]")
        print(f"total matched flux ratio = {rf[good].sum()/tf[good].sum():.3f}")
        # brightest-source honesty check
        bi = int(np.argmax(tf))
        print(f"brightest matched star: truth flux {tf[bi]:.3e}, "
              f"recon flux {rf[bi]:.3e} (ratio {rf[bi]/tf[bi]:.3f}), "
              f"pos err {dists[bi]:.2f} px")

    # CSV
    csv_path = os.path.join(args.out_dir,
                            os.path.splitext(os.path.basename(args.recon))[0]
                            + "_photometry.csv")
    with open(csv_path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["truth_x", "truth_y", "recon_x", "recon_y", "dist_px",
                     "truth_flux", "recon_flux", "flux_ratio"])
        for k, (ti, ri, d) in enumerate(pairs):
            wr.writerow([f"{t_xy[ti][0]:.2f}", f"{t_xy[ti][1]:.2f}",
                         f"{r_xy[ri][0]:.2f}", f"{r_xy[ri][1]:.2f}", f"{d:.3f}",
                         f"{tf[k]:.4e}", f"{rf[k]:.4e}",
                         f"{(rf[k]/tf[k]) if tf[k]>0 else np.nan:.3f}"])

    # figure: overlay + flux scatter
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    for a, img, title in [(ax[0], truth, "truth + detections"),
                          (ax[1], recon, "recon + detections")]:
        lw = max(t_std, 1e-30)
        a.imshow(img, origin="lower", cmap="gray",
                 norm=AsinhNorm(linear_width=lw, vmin=float(img.min()),
                                vmax=float(img.max())))
        if len(t_xy):
            a.scatter(t_xy[:, 0], t_xy[:, 1], s=90, facecolors="none",
                      edgecolors="lime", lw=1.2, label="truth")
        if len(r_xy):
            a.scatter(r_xy[:, 0], r_xy[:, 1], marker="+", c="red", s=60,
                      lw=1.0, label="recon")
        a.set_title(title, fontsize=10); a.axis("off"); a.legend(fontsize=8, loc="upper right")
    if n_m:
        good = np.isfinite(flux_ratio) & (tf > 0)
        ax[2].loglog(tf[good], rf[good], "o", ms=5, alpha=0.7)
        lo = min(tf[good].min(), rf[good].min()); hi = max(tf[good].max(), rf[good].max())
        ax[2].plot([lo, hi], [lo, hi], "k--", lw=1, label="1:1")
        ax[2].set_xlabel("truth aperture flux"); ax[2].set_ylabel("recon aperture flux")
        ax[2].set_title(f"flux: median ratio {np.median(flux_ratio[good]):.2f}", fontsize=10)
        ax[2].legend(fontsize=8)
    fig.tight_layout()
    png = os.path.join(args.out_dir,
                       os.path.splitext(os.path.basename(args.recon))[0]
                       + "_photometry.png")
    fig.savefig(png, dpi=130); plt.close(fig)
    print(f"\nwrote {csv_path}")
    print(f"wrote {png}")


if __name__ == "__main__":
    main()

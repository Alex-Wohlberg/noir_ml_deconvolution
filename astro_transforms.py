"""
astro_transforms.py

Single source of truth for the asinh normalization used by:
  - StarPatchDataset (training the diffusion prior)
  - the PnP-PGD proximal wrapper (inference)
  - any evaluation / visualization code

The transform pair:
    z = asinh(x / b) / A          (flux -> network space)
    x = b * sinh(z * A)           (network space -> flux, exact inverse)

  b : softening scale. Linear below b, logarithmic above. Set from the
      sharp-patch background noise sigma (times a small factor) so the
      noise floor passes through un-amplified while stars are compressed.
  A : range constant. Set so the 99.9th percentile of the BLURRY pixels
      maps to z = 1.0 -- the blurry domain bounds the dynamic range the
      PnP loop will push through this transform, so the whole loop stays
      inside the range the prior was trained on.

Negative background pixels pass through untouched (asinh is odd); there is
NO lower clip at zero. The upper clip is the pathological-outlier guard
only; the lower bound is a generous -5 sigma-equivalent floor.

Works on numpy arrays and torch tensors alike.
"""

import json
import math

import numpy as np

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _xp(x):
    """Return the array module matching x (numpy or torch)."""
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return torch
    return np


class AsinhTransform:
    def __init__(self, b: float, A: float, noise_sigma: float,
                 z_ceil: float = 1.05, floor_nsigma: float = 5.0,
                 net_scale: float = None):
        if b <= 0 or A <= 0:
            raise ValueError("b and A must be positive")
        self.b = float(b)
        self.A = float(A)
        self.noise_sigma = float(noise_sigma)
        self.z_ceil = float(z_ceil)
        # generous negative floor: -floor_nsigma * noise_sigma in flux,
        # mapped through the transform (NOT zero -- background noise must
        # keep its negative half or faint photometry biases upward)
        self.z_floor = float(
            math.asinh(-floor_nsigma * noise_sigma / b) / A
        )
        # Net-space affine on top of z:  x_net = z / net_scale - 1.
        # Why: z-space sharp data has std ~0.09 against unit diffusion
        # noise, so "predict blank" is loss-optimal and the prior collapses
        # to noise (the flat_cnn_stars.pt failure). With s = 0.20 the
        # background (z = 0) sits exactly at -1, std rises to ~0.46, and
        # the signal emerges a quarter of the way into the reverse chain
        # instead of the last 5%. None = legacy raw-z behavior (old
        # checkpoints trained without the affine).
        self.net_scale = None if net_scale is None else float(net_scale)

    # ------------------------------------------------------------------
    # The transform pair
    # ------------------------------------------------------------------
    def to_z(self, x):
        """Flux -> network space, with outlier guard clipping."""
        xp = _xp(x)
        z = xp.arcsinh(x / self.b) / self.A
        return xp.clip(z, self.z_floor, self.z_ceil)

    def to_flux(self, z):
        """z-space -> flux. Exact inverse of to_z inside the clip range."""
        xp = _xp(z)
        return self.b * xp.sinh(z * self.A)

    # ------------------------------------------------------------------
    # Net-space affine (z <-> what the network actually trains on)
    # ------------------------------------------------------------------
    def z_to_net(self, z):
        """z -> network space. Identity when net_scale is None (legacy)."""
        if self.net_scale is None:
            return z
        return z / self.net_scale - 1.0

    def net_to_z(self, x):
        """Network space -> z. Exact inverse of z_to_net."""
        if self.net_scale is None:
            return x
        return (x + 1.0) * self.net_scale

    def to_net(self, x_flux):
        """Flux -> network space (asinh + affine + clip)."""
        return self.z_to_net(self.to_z(x_flux))

    def net_to_flux(self, x):
        """Network space -> flux. Exact inverse of to_net inside the clips."""
        return self.to_flux(self.net_to_z(x))

    @property
    def net_floor(self):
        return float(self.z_to_net(self.z_floor))

    @property
    def net_ceil(self):
        return float(self.z_to_net(self.z_ceil))

    # ------------------------------------------------------------------
    # PnP glue
    # ------------------------------------------------------------------
    def sigma_flux_to_z(self, sigma_flux: float) -> float:
        """Map a flux-space noise level to z-space, in the linear regime.

        Residual noise in the PnP iterate lives near the background, where
        the transform slope is d(z)/d(x)|_{x~0} = 1 / (b * A).
        """
        return float(sigma_flux) / (self.b * self.A)

    def sigma_flux_to_model(self, sigma_flux: float) -> float:
        """Flux-space noise level -> MODEL-space, i.e. the space the
        diffusion prior trains and denoises in (net space when net_scale
        is set, raw z otherwise). This is the sigma to hand to
        Schedule.sigma_to_t -- using sigma_flux_to_z with an affine-trained
        model picks timesteps 1/net_scale (~5x) too early.
        """
        s = 1.0 if self.net_scale is None else self.net_scale
        return self.sigma_flux_to_z(sigma_flux) / s

    # ------------------------------------------------------------------
    # Derivation from data
    # ------------------------------------------------------------------
    @classmethod
    def derive(cls, sharp: np.ndarray, blurry: np.ndarray,
               b_factor: float = 1.5, pct: float = 99.9,
               z_ceil: float = 1.05, floor_nsigma: float = 5.0,
               net_scale: float = None):
        """Derive (b, A) from clean, flux-consistent patch stacks.

        IMPORTANT: run this only AFTER the blurry/sharp flux inconsistency
        (plot 07) is fixed -- constants derived from inconsistent data get
        baked into the trained prior.
        """
        noise_sigma = _sigma_clipped_std(sharp)
        b = b_factor * noise_sigma
        p = float(np.percentile(np.asarray(blurry).ravel(), pct))
        if p <= 0:
            raise ValueError(f"blurry {pct}th percentile is {p}; data looks wrong")
        A = math.asinh(p / b)
        return cls(b=b, A=A, noise_sigma=noise_sigma,
                   z_ceil=z_ceil, floor_nsigma=floor_nsigma,
                   net_scale=net_scale)

    # ------------------------------------------------------------------
    # Persistence + drift protection
    # ------------------------------------------------------------------
    def to_dict(self) -> dict:
        return {"b": self.b, "A": self.A, "noise_sigma": self.noise_sigma,
                "z_ceil": self.z_ceil, "z_floor": self.z_floor,
                "net_scale": self.net_scale}

    @classmethod
    def from_dict(cls, d: dict) -> "AsinhTransform":
        t = cls(b=d["b"], A=d["A"], noise_sigma=d["noise_sigma"],
                z_ceil=d.get("z_ceil", 1.05),
                net_scale=d.get("net_scale"))  # absent in old JSONs -> None
        # trust the stored floor over recomputation (guards param drift)
        t.z_floor = d["z_floor"]
        return t

    def assert_matches(self, d: dict, rtol: float = 1e-6):
        """Fail loudly if checkpoint constants differ from this instance."""
        for k in ("b", "A", "z_floor", "z_ceil"):
            mine, theirs = getattr(self, k), d[k]
            if not math.isclose(mine, theirs, rel_tol=rtol):
                raise ValueError(
                    f"AsinhTransform mismatch on '{k}': {mine} vs checkpoint "
                    f"{theirs}. Training and inference must share one transform."
                )
        mine, theirs = self.net_scale, d.get("net_scale")
        if (mine is None) != (theirs is None) or (
                mine is not None
                and not math.isclose(mine, theirs, rel_tol=rtol)):
            raise ValueError(
                f"AsinhTransform mismatch on 'net_scale': {mine} vs "
                f"checkpoint {theirs}. A prior trained with the net-space "
                f"affine cannot be driven through raw-z constants (or vice "
                f"versa).")

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AsinhTransform":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def __repr__(self):
        net = ("raw-z (legacy)" if self.net_scale is None
               else f"net_scale={self.net_scale:.3g}, "
                    f"net_range=[{self.net_floor:.2f}, {self.net_ceil:.2f}]")
        return (f"AsinhTransform(b={self.b:.4g}, A={self.A:.4g}, "
                f"noise_sigma={self.noise_sigma:.4g}, "
                f"z_range=[{self.z_floor:.3f}, {self.z_ceil:.3f}], {net})")


def _sigma_clipped_std(a, iters=5, k=3.0):
    a = np.asarray(a).ravel().copy()
    for _ in range(iters):
        med, std = np.median(a), np.std(a)
        keep = np.abs(a - med) < k * std
        if keep.sum() == 0:
            return float(std)
        a = a[keep]
    return float(np.std(a))
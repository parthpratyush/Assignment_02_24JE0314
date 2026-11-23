"""Grid search and bounded nonlinear refinement for spatial Orr-Sommerfeld eigenvalues."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter
from scipy.optimize import least_squares

from .compound_matrix import CompoundMatrixSolver


@dataclass
class RootResult:
    alpha: complex
    residual: complex
    grid_seed: complex
    iterations: int

    @property
    def stable(self) -> bool:
        return self.alpha.imag > 0.0


@dataclass
class GridSearchResult:
    alpha_r: np.ndarray
    alpha_i: np.ndarray
    dispersion: np.ndarray
    roots: list[RootResult]


def grid_search(
    solver: CompoundMatrixSolver,
    alpha_r: np.ndarray,
    alpha_i: np.ndarray,
    *,
    scan_step: float = 0.02,
    refine_step: float = 0.005,
    residual_tolerance: float = 2e-7,
    max_candidates: int = 4,
    max_roots: int = 8,
    progress: bool = False,
) -> GridSearchResult:
    """Find zero-contour intersections in the complex-alpha plane."""
    alpha_r = np.asarray(alpha_r, dtype=float)
    alpha_i = np.asarray(alpha_i, dtype=float)
    if alpha_r.ndim != 1 or alpha_i.ndim != 1 or len(alpha_r) < 3 or len(alpha_i) < 3:
        raise ValueError("alpha_r and alpha_i must be one-dimensional grids with at least 3 points")
    if np.any(np.diff(alpha_r) <= 0) or np.any(np.diff(alpha_i) <= 0):
        raise ValueError("The alpha grids must be strictly increasing")

    values = np.empty((len(alpha_i), len(alpha_r)), dtype=np.complex128)
    for j, imag_part in enumerate(alpha_i):
        for i, real_part in enumerate(alpha_r):
            values[j, i] = solver.dispersion(real_part + 1j * imag_part, step=scan_step)
        if progress and ((j + 1) % 5 == 0 or j + 1 == len(alpha_i)):
            print(f"  Grid rows completed: {j + 1}/{len(alpha_i)}")

    magnitude = np.abs(values)
    local_minima = magnitude <= minimum_filter(magnitude, size=3, mode="nearest") * (1.0 + 1e-12)
    candidates = [
        (float(magnitude[j, i]), float(alpha_r[i]), float(alpha_i[j]))
        for j, i in zip(*np.where(local_minima))
    ]
    candidates.sort(key=lambda item: item[0])

    roots: list[RootResult] = []
    ar_min, ar_max = float(alpha_r[0]), float(alpha_r[-1])
    ai_min, ai_max = float(alpha_i[0]), float(alpha_i[-1])

    def residual(vector: np.ndarray) -> np.ndarray:
        value = solver.dispersion(complex(float(vector[0]), float(vector[1])), step=refine_step)
        return np.asarray([value.real, value.imag], dtype=float)

    tried = 0
    for _magnitude, real_seed, imag_seed in candidates[:max_candidates]:
        seed = complex(real_seed, imag_seed)
        if any(abs(seed - existing.alpha) < 2e-3 for existing in roots):
            continue
        tried += 1
        try:
            refined = least_squares(
                residual,
                np.asarray([real_seed, imag_seed], dtype=float),
                bounds=(np.asarray([ar_min, ai_min]), np.asarray([ar_max, ai_max])),
                method="trf",
                x_scale="jac",
                ftol=1e-11,
                xtol=1e-11,
                gtol=1e-11,
                max_nfev=100,
            )
        except (ValueError, FloatingPointError, OverflowError):
            continue
        alpha = complex(float(refined.x[0]), float(refined.x[1]))
        if not refined.success:
            continue
        if not np.isfinite(alpha.real) or not np.isfinite(alpha.imag):
            continue
        wall_residual = solver.dispersion(alpha, step=refine_step)
        if abs(wall_residual) > residual_tolerance:
            continue
        if any(abs(alpha - existing.alpha) < 2e-4 for existing in roots):
            continue
        roots.append(
            RootResult(alpha, wall_residual, seed, int(getattr(refined, "nfev", 0)))
        )
        if len(roots) >= max_roots:
            break

    roots.sort(key=lambda item: (item.alpha.real, item.alpha.imag))
    if progress:
        print(f"  Refined {len(roots)} distinct root(s) from {tried} grid candidate(s).")
    return GridSearchResult(alpha_r, alpha_i, values, roots)

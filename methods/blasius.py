"""Classical Blasius base-flow profile in displacement-thickness coordinates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson, solve_ivp
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq


@dataclass
class BlasiusProfile:
    """Return U(y_tilde) and its second derivative for the Blasius layer.

    The similarity equation is
        f''' + 0.5*f*f'' = 0,
    with f(0)=f'(0)=0 and f'(infinity)=1.

    The displacement thickness is
        delta* = integral_0^infinity (1 - f'(eta)) d eta.
    Therefore eta = delta*_factor * y_tilde, where y_tilde = y/delta*.
    """

    y_max: float = 12.0
    profile_points: int = 6001

    def __post_init__(self) -> None:
        if self.y_max <= 0:
            raise ValueError("y_max must be positive")
        if self.profile_points < 1000:
            raise ValueError("profile_points must be at least 1000")

        eta_max = max(25.0, 1.8 * self.y_max + 5.0)

        def blasius_rhs(_eta: float, state: np.ndarray) -> list[float]:
            f, _fp, fpp = state
            return [_fp, fpp, -0.5 * f * fpp]

        def wall_shear_residual(fpp0: float) -> float:
            trial = solve_ivp(
                blasius_rhs,
                (0.0, eta_max),
                (0.0, 0.0, fpp0),
                method="DOP853",
                rtol=2e-10,
                atol=2e-12,
            )
            if not trial.success:
                raise RuntimeError(f"Blasius shooting integration failed: {trial.message}")
            return float(trial.y[1, -1] - 1.0)

        wall_shear = brentq(wall_shear_residual, 0.2, 0.5, xtol=1e-13)
        eta = np.linspace(0.0, eta_max, self.profile_points)
        solution = solve_ivp(
            blasius_rhs,
            (0.0, eta_max),
            (0.0, 0.0, wall_shear),
            method="DOP853",
            t_eval=eta,
            dense_output=True,
            rtol=2e-12,
            atol=2e-14,
        )
        if not solution.success:
            raise RuntimeError(f"Blasius profile integration failed: {solution.message}")

        f, fp, fpp = solution.y
        delta_factor = float(simpson(1.0 - fp, x=eta))
        if not np.isfinite(delta_factor) or delta_factor <= 0:
            raise RuntimeError("Could not determine a positive Blasius displacement thickness")

        self.wall_shear = float(wall_shear)
        self.delta_factor = delta_factor
        self.eta_max = float(eta_max)
        self.eta = eta
        self._f = CubicSpline(eta, f)
        self._fp = CubicSpline(eta, fp)
        self._fpp = CubicSpline(eta, fpp)

        self.y_tilde = np.linspace(0.0, self.y_max, self.profile_points)
        eta_scaled = self.delta_factor * self.y_tilde
        f_scaled = solution.sol(eta_scaled)
        self.U_values = np.asarray(f_scaled[1], dtype=float)
        fppp_scaled = -0.5 * f_scaled[0] * f_scaled[2]
        self.Upp_values = self.delta_factor**2 * np.asarray(fppp_scaled, dtype=float)
        self._U_spline = CubicSpline(self.y_tilde, self.U_values)
        self._Upp_spline = CubicSpline(self.y_tilde, self.Upp_values)

    def U(self, y_tilde: float | np.ndarray) -> float | np.ndarray:
        """Dimensionless streamwise base velocity U/U_e."""
        return self._U_spline(y_tilde)

    def Upp(self, y_tilde: float | np.ndarray) -> float | np.ndarray:
        """Second derivative d²U/d(y_tilde)²."""
        return self._Upp_spline(y_tilde)

    def f(self, eta: float | np.ndarray) -> float | np.ndarray:
        return self._f(eta)

    def fp(self, eta: float | np.ndarray) -> float | np.ndarray:
        return self._fp(eta)

    def fpp(self, eta: float | np.ndarray) -> float | np.ndarray:
        return self._fpp(eta)

"""Six-dimensional compound-matrix formulation of the 2-D Orr-Sommerfeld ODE."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .blasius import BlasiusProfile


@dataclass
class CompoundMatrixSolver:
    r"""Integrate the six compound variables from the free stream to the wall.

    For complex alpha and real omega_0, the 2-D Orr-Sommerfeld equation is

      (D²-alpha²)² phi = i Re [(alpha U - omega_0)(D²-alpha²)phi
                              - alpha U'' phi].

    The six variables are the 2x2 minors of the two-dimensional decaying
    solution subspace. Their first-order system is Eqs. (2.30)-(2.35).
    """

    reynolds: float = 1500.0
    omega0: float = 0.1
    y_max: float = 12.0
    profile: BlasiusProfile | None = None
    _integration_cache: dict[int, tuple[np.ndarray, ...]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if self.reynolds <= 0:
            raise ValueError("Reynolds number must be positive")
        if self.y_max <= 0:
            raise ValueError("y_max must be positive")
        if self.profile is None:
            self.profile = BlasiusProfile(y_max=self.y_max)
        elif self.profile.y_max < self.y_max:
            raise ValueError("The supplied Blasius profile must cover y_max")

    @staticmethod
    def free_stream_decay_root(alpha: complex, reynolds: float, omega0: float) -> complex:
        """Return Q = sqrt(alpha² + i Re(alpha-omega_0)) with Re(Q)>0."""
        q = complex(np.sqrt(alpha * alpha + 1j * reynolds * (alpha - omega0)))
        if q.real < 0.0:
            q = -q
        if q.real <= 0.0:
            raise ValueError("The decaying free-stream branch requires Re(Q) > 0")
        return q

    @staticmethod
    def initial_compound_state(alpha: complex, q: complex) -> np.ndarray:
        """Scaled far-field initial values (Eqs. 2.36-2.41)."""
        a = complex(alpha)
        return np.asarray(
            [
                a - q,
                q**2 - a**2,
                a**3 - q**3,
                a**2 * q - a * q**2,
                a * q**3 - a**3 * q,
                a**3 * q**2 - a**2 * q**3,
            ],
            dtype=np.complex128,
        )

    def _profile_grid(self, step: float) -> tuple[np.ndarray, ...]:
        n_steps = int(np.ceil(self.y_max / step))
        if n_steps < 1:
            raise ValueError("Integration step must be smaller than y_max")
        if n_steps not in self._integration_cache:
            y_nodes = np.linspace(self.y_max, 0.0, n_steps + 1)
            y_mid = 0.5 * (y_nodes[:-1] + y_nodes[1:])
            assert self.profile is not None
            grid = (
                y_nodes,
                y_mid,
                np.asarray(self.profile.U(y_nodes), dtype=float),
                np.asarray(self.profile.Upp(y_nodes), dtype=float),
                np.asarray(self.profile.U(y_mid), dtype=float),
                np.asarray(self.profile.Upp(y_mid), dtype=float),
            )
            self._integration_cache[n_steps] = grid
        return self._integration_cache[n_steps]

    def dispersion(
        self,
        alpha: complex,
        step: float = 0.01,
        *,
        return_state: bool = False,
    ) -> complex | tuple[complex, np.ndarray]:
        """Compute the scaled wall dispersion value y_1(0; alpha, omega_0, Re)."""
        alpha = complex(alpha)
        if alpha.real <= 0.0:
            raise ValueError("Spatial wavenumber search requires alpha_r > 0")
        if step <= 0.0:
            raise ValueError("Integration step must be positive")

        q = self.free_stream_decay_root(alpha, self.reynolds, self.omega0)
        state = self.initial_compound_state(alpha, q)
        y_nodes, _y_mid, u_nodes, upp_nodes, u_mid, upp_mid = self._profile_grid(step)
        h = -self.y_max / (len(y_nodes) - 1)
        re = self.reynolds
        omega0 = self.omega0
        a2 = alpha * alpha
        a4 = a2 * a2

        def rhs(z: np.ndarray, u: float, upp: float) -> np.ndarray:
            b1 = 2.0 * a2 + 1j * re * (alpha * u - omega0)
            b2 = a4 + 1j * re * alpha * upp + 1j * re * a2 * (alpha * u - omega0)
            y1, y2, y3, y4, y5, y6 = z
            return np.asarray(
                [
                    y2,
                    y3 + y4,
                    b1 * y2 + y5,
                    y5,
                    y6 + b1 * y4 + b2 * y1,
                    b2 * y2,
                ],
                dtype=np.complex128,
            )

        for index in range(len(y_nodes) - 1):
            k1 = rhs(state, u_nodes[index], upp_nodes[index])
            k2 = rhs(state + 0.5 * h * k1, u_mid[index], upp_mid[index])
            k3 = rhs(state + 0.5 * h * k2, u_mid[index], upp_mid[index])
            k4 = rhs(state + h * k3, u_nodes[index + 1], upp_nodes[index + 1])
            state = state + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

            scale = float(np.max(np.abs(state)))
            if not np.isfinite(scale) or scale == 0.0:
                raise FloatingPointError("Compound integration became non-finite; reduce step or adjust search domain")
            state /= scale

        value = complex(state[0])
        return (value, state.copy()) if return_state else value

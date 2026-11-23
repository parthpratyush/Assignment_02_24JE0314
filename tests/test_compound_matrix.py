"""Regression tests against the benchmark roots printed in the notes."""

import numpy as np

from methods.blasius import BlasiusProfile
from methods.compound_matrix import CompoundMatrixSolver

EXPECTED_ROOTS = np.asarray([
    0.13301162 + 0.21791196j,
    0.17675906 + 0.12104521j,
    0.29373724 - 0.00703994013j,
])


def test_analytical_free_stream_initial_values_are_finite():
    alpha = 0.3 - 0.01j
    solver = CompoundMatrixSolver(reynolds=1500.0, omega0=0.1, y_max=12.0)
    q = solver.free_stream_decay_root(alpha, 1500.0, 0.1)
    y0 = solver.initial_compound_state(alpha, q)
    assert y0.shape == (6,)
    assert np.all(np.isfinite(y0))
    assert q.real > 0.0


def test_reference_roots_satisfy_wall_dispersion_condition():
    profile = BlasiusProfile(y_max=12.0)
    solver = CompoundMatrixSolver(1500.0, 0.1, 12.0, profile)
    for alpha in EXPECTED_ROOTS:
        assert abs(solver.dispersion(alpha, step=0.002)) < 2e-6

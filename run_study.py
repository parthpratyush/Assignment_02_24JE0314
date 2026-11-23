"""Run the Orr-Sommerfeld compound-matrix grid search and generate figures."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from methods.blasius import BlasiusProfile
from methods.compound_matrix import CompoundMatrixSolver
from methods.root_finding import GridSearchResult, grid_search

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
RESULTS = ROOT / "results.csv"
GRID_FILE = ROOT / "dispersion_grid.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--re", type=float, default=1500.0, help="displacement-thickness Reynolds number")
    parser.add_argument("--omega0", type=float, default=0.1, help="non-dimensional circular frequency")
    parser.add_argument("--y-max", type=float, default=12.0, help="finite free-stream integration location")
    parser.add_argument("--alpha-r-min", type=float, default=0.08)
    parser.add_argument("--alpha-r-max", type=float, default=0.50)
    parser.add_argument("--alpha-i-min", type=float, default=-0.06)
    parser.add_argument("--alpha-i-max", type=float, default=0.50)
    parser.add_argument("--alpha-r-points", type=int, default=25)
    parser.add_argument("--alpha-i-points", type=int, default=31)
    parser.add_argument("--scan-step", type=float, default=0.02)
    parser.add_argument("--refine-step", type=float, default=0.005)
    return parser.parse_args()


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f"{filename}.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIGURES / f"{filename}.svg", bbox_inches="tight")
    plt.close(fig)


def write_results(result: GridSearchResult, reynolds: float, omega0: float) -> None:
    with RESULTS.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "Re_delta_star", "omega0", "alpha_r", "alpha_i", "abs_wall_residual", "spatial_behavior"])
        for index, mode in enumerate(result.roots, start=1):
            behavior = "spatially unstable (growth)" if mode.alpha.imag < 0 else "spatially stable (decay)"
            writer.writerow([index, reynolds, omega0, f"{mode.alpha.real:.10f}", f"{mode.alpha.imag:.10f}", f"{abs(mode.residual):.4e}", behavior])


def plot_dispersion_contours(result: GridSearchResult, reynolds: float, omega0: float) -> None:
    ar, ai = np.meshgrid(result.alpha_r, result.alpha_i)
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    ax.contour(ar, ai, result.dispersion.real, levels=[0.0], colors=["tab:red"], linewidths=2.0)
    ax.contour(ar, ai, result.dispersion.imag, levels=[0.0], colors=["black"], linewidths=2.0, linestyles="--")
    ax.legend(handles=[
        Line2D([0], [0], color="tab:red", linewidth=2, label=r"$\Re[y_1(0)] = 0$"),
        Line2D([0], [0], color="black", linewidth=2, linestyle="--", label=r"$\Im[y_1(0)] = 0$"),
    ], loc="best")
    for number, mode in enumerate(result.roots, start=1):
        ax.plot(mode.alpha.real, mode.alpha.imag, "o", markersize=7, markeredgecolor="black")
        ax.annotate(f"Mode {number}", (mode.alpha.real, mode.alpha.imag), xytext=(7, 7), textcoords="offset points")
    ax.set_xlabel(r"Real wavenumber, $\alpha_r$")
    ax.set_ylabel(r"Imaginary wavenumber, $\alpha_i$")
    ax.set_title(rf"Compound-matrix dispersion contours ($Re_{{\delta^*}}={reynolds:g}$, $\omega_0={omega0:g}$)")
    ax.grid(alpha=0.25)
    save_figure(fig, "dispersion_contours")


def plot_dispersion_magnitude(result: GridSearchResult, reynolds: float, omega0: float) -> None:
    ar, ai = np.meshgrid(result.alpha_r, result.alpha_i)
    magnitude = np.log10(np.maximum(np.abs(result.dispersion), 1e-14))
    fig, ax = plt.subplots(figsize=(8.2, 6.2))
    image = ax.contourf(ar, ai, magnitude, levels=28)
    fig.colorbar(image, ax=ax, label=r"$\log_{10}|y_1(0)|$ (scaled)")
    for number, mode in enumerate(result.roots, start=1):
        ax.plot(mode.alpha.real, mode.alpha.imag, "wo", markeredgecolor="black", markersize=7)
        ax.annotate(f"{number}", (mode.alpha.real, mode.alpha.imag), xytext=(6, 6), textcoords="offset points")
    ax.set_xlabel(r"Real wavenumber, $\alpha_r$")
    ax.set_ylabel(r"Imaginary wavenumber, $\alpha_i$")
    ax.set_title(rf"Scaled dispersion residual ($Re_{{\delta^*}}={reynolds:g}$, $\omega_0={omega0:g}$)")
    save_figure(fig, "dispersion_magnitude")


def plot_base_flow(profile: BlasiusProfile) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    axes[0].plot(profile.y_tilde, profile.U_values, linewidth=2)
    axes[0].set_xlabel(r"$\widetilde y = y/\delta^*$")
    axes[0].set_ylabel(r"$U/U_e$")
    axes[0].set_title("Blasius base velocity")
    axes[0].grid(alpha=0.3)
    axes[1].plot(profile.y_tilde, profile.Upp_values, linewidth=2)
    axes[1].axhline(0.0, linewidth=0.8)
    axes[1].set_xlabel(r"$\widetilde y = y/\delta^*$")
    axes[1].set_ylabel(r"$d^2(U/U_e)/d\widetilde y^2$")
    axes[1].set_title("Second derivative of base velocity")
    axes[1].grid(alpha=0.3)
    save_figure(fig, "blasius_base_flow")


def plot_spatial_behavior(result: GridSearchResult) -> None:
    if not result.roots:
        return
    indices = np.arange(1, len(result.roots) + 1)
    alpha_i = np.asarray([mode.alpha.imag for mode in result.roots])
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.bar(indices, alpha_i)
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_xticks(indices)
    ax.set_xlabel("Mode")
    ax.set_ylabel(r"$\alpha_i$")
    ax.set_title("Spatial growth / decay indicator")
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, "spatial_growth_indicator")


def main() -> None:
    args = parse_args()
    if args.alpha_r_points < 3 or args.alpha_i_points < 3:
        raise SystemExit("At least three grid points are required in each direction")

    print("Orr-Sommerfeld spatial stability: compound matrix method")
    print(f"Re_delta* = {args.re:g}, omega0 = {args.omega0:g}, y_max = {args.y_max:g}")
    profile = BlasiusProfile(y_max=args.y_max)
    print(f"Blasius f''(0) = {profile.wall_shear:.12f}")
    print(f"Blasius displacement-thickness factor = {profile.delta_factor:.12f}")

    solver = CompoundMatrixSolver(args.re, args.omega0, args.y_max, profile)
    alpha_r = np.linspace(args.alpha_r_min, args.alpha_r_max, args.alpha_r_points)
    alpha_i = np.linspace(args.alpha_i_min, args.alpha_i_max, args.alpha_i_points)
    result = grid_search(solver, alpha_r, alpha_i, scan_step=args.scan_step, refine_step=args.refine_step, progress=True)

    write_results(result, args.re, args.omega0)
    np.savez_compressed(GRID_FILE, alpha_r=result.alpha_r, alpha_i=result.alpha_i, dispersion=result.dispersion, reynolds=args.re, omega0=args.omega0, y_max=args.y_max)
    plot_dispersion_contours(result, args.re, args.omega0)
    plot_dispersion_magnitude(result, args.re, args.omega0)
    plot_base_flow(profile)
    plot_spatial_behavior(result)

    print("\nRefined spatial eigenvalues")
    print("Mode       alpha_r             alpha_i          |y1(0)|       behaviour")
    print("-" * 82)
    for index, mode in enumerate(result.roots, start=1):
        behaviour = "growth" if mode.alpha.imag < 0 else "decay"
        print(f"{index:<5d} {mode.alpha.real: .10f}  {mode.alpha.imag: .10f}  {abs(mode.residual):.3e}   {behaviour}")
    if not result.roots:
        print("No roots were refined. Expand the alpha search box or refine the grid.")
    print("\nGenerated: results.csv, dispersion_grid.npz, and figures/*.png + *.svg")


if __name__ == "__main__":
    main()

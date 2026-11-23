from __future__ import annotations
import argparse, csv
from pathlib import Path
import numpy as np
from methods.blasius import BlasiusProfile
from methods.root_finding import GridSearchResult, RootResult
from run_study import plot_base_flow, plot_dispersion_contours, plot_dispersion_magnitude, plot_spatial_behavior

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--grid', type=Path, default=ROOT / 'dispersion_grid.npz')
    args = parser.parse_args()
    if not args.grid.exists():
        raise SystemExit('dispersion_grid.npz is missing; run python run_study.py first.')
    data = np.load(args.grid)
    roots = []
    with (ROOT / 'results.csv').open(newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            a = complex(float(row['alpha_r']), float(row['alpha_i']))
            roots.append(RootResult(a, 0j, a, 0))
    result = GridSearchResult(data['alpha_r'], data['alpha_i'], data['dispersion'], roots)
    profile = BlasiusProfile(y_max=float(data['y_max']))
    plot_dispersion_contours(result, float(data['reynolds']), float(data['omega0']))
    plot_dispersion_magnitude(result, float(data['reynolds']), float(data['omega0']))
    plot_base_flow(profile)
    plot_spatial_behavior(result)
    print('Figures regenerated in figures/.')

if __name__ == '__main__':
    main()

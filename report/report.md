# Numerical Study of the 2-D Orr-Sommerfeld Equation using the Compound Matrix Method

**Student:** Pratyush Singh Parmar  
**Roll No.:** 24JE0314  
**Institute:** IIT (ISM) Dhanbad

## Objective
Find the spatial eigenvalues of a Blasius boundary layer. The complex streamwise wavenumber is alpha = alpha_r + i alpha_i; the non-dimensional circular frequency omega0 and Reynolds number are real.

## Governing equation
```text
(D^2 - alpha^2)^2 phi = i Re_delta* [(alpha U - omega0)(D^2 - alpha^2)phi - alpha U'' phi]
```
The wall conditions are phi(0)=phi'(0)=0. In the free stream, the two decaying modes behave as exp(-alpha*y_tilde) and exp(-Q*y_tilde), with Q=sqrt(alpha^2+i Re_delta*(alpha-omega0)) and Re(Q)>0.

## Compound-matrix system
```text
y1' = y2                 y2' = y3 + y4
y3' = b1*y2 + y5         y4' = y5
y5' = y6 + b1*y4 + b2*y1 y6' = b2*y2

b1 = 2*alpha^2 + i*Re_delta*(alpha*U - omega0)
b2 = alpha^4 + i*Re_delta*alpha*U'' + i*Re_delta*alpha^2*(alpha*U - omega0)
```
Far-field initial conditions at y_tilde=y_max:
```text
y1=alpha-Q; y2=Q^2-alpha^2; y3=alpha^3-Q^3
y4=alpha^2*Q-alpha*Q^2; y5=alpha*Q^3-alpha^3*Q
y6=alpha^3*Q^2-alpha^2*Q^3
```
The six-equation system is integrated backward with RK4 and common positive rescaling. The wall dispersion relation is y1(0)=0.

## Reference results

Run case: Re_delta*=1500, omega0=0.1, y_max=12.

| Mode | alpha_r | alpha_i | abs wall residual | Behaviour |
|---:|---:|---:|---:|---|
| 1 | 0.13301166 | 0.21791206 | 2.171e-11 | Decay |
| 2 | 0.17675903 | 0.12104523 | 4.579e-15 | Decay |
| 3 | 0.29373736 | -0.00703999 | 4.878e-10 | Growth |

Spatial convention: alpha_i<0 gives downstream growth; alpha_i>0 gives downstream decay. The three modes agree closely with the reference table in the supplied Orr-Sommerfeld lecture PDF.

## Reproducibility

`python run_study.py` generates `results.csv`, `dispersion_grid.npz`, and four plots in `figures/`. `python build_report.py` creates `REPORT.pdf`. `pytest -q` runs the numerical regression tests.

## Assumptions

The notes do not contain a tabulated Blasius velocity profile; the code computes the classical Blasius similarity solution and converts it to displacement-thickness coordinates. The infinite domain is truncated at y_max=12 for the reference calculation.

"""Partial identification of a win rate with missing results, via a shadow variable (binary outcome).

Adapted from the partial-identification-by-LP approach (Simchi-Levi et al.): a noisy predictor Z
is used as a *shadow variable* to bound E[Y] when some outcomes are missing not at random.

    Y in {0,1}  outcome (team A won)                  -- missing when R = 0
    R in {0,1}  outcome observed?
    X           observed covariates (discrete bucket, e.g. Elo-difference bin)
    Z           shadow variable (discrete bin of our model's predicted win probability)

Assumption:  Z ⟂ R | (Y, X)   -- given who won (and X), whether the result went missing doesn't
depend on what the model predicted. The model does NOT have to be accurate.

Unknowns: q[y, x] = P(Y=y, X=x, R=0). The shadow link gives, for every (z, x):
    sum_y P(Z=z | Y=y, X=x, R=1) * q[y, x] = P(Z=z, X=x, R=0)
Two LPs (min / max of E[Y] = P(Y=1, R=1) + sum_x q[1, x]) give sharp bounds. Finite samples make
the equalities slightly infeasible, so (localised penalisation) we first find the minimum total
slack, then allow slack up to that minimum + a tolerance shrinking like 1/sqrt(n).
With binary Y there are only 2 unknowns per X-bucket, so an informative Z usually pins E[Y] down
almost to a point.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def worst_case_bounds(y, r):
    """No assumptions: every missing result could be a win or a loss."""
    y, r = np.asarray(y, float), np.asarray(r, bool)
    p_obs_win = np.mean(r & (y == 1))
    return p_obs_win, p_obs_win + np.mean(~r)


def shadow_bounds(y, r, z, x, tol_scale: float = 1.0):
    """Sharp bounds on P(Y=1) using shadow variable z (int bins) and covariate x (int bins)."""
    y, r = np.asarray(y, float), np.asarray(r, bool)
    z, x = np.asarray(z, int), np.asarray(x, int)
    n = len(y)
    Zs, Xs = np.unique(z), np.unique(x)
    nz, nx = len(Zs), len(Xs)
    zi = {v: i for i, v in enumerate(Zs)}
    xi = {v: i for i, v in enumerate(Xs)}
    nq = 2 * nx                      # q[y, x], index y*nx + x
    ns = 2 * nz * nx                 # slack +/- per (z, x) equation
    A_eq, b_eq = [], []
    for xv in Xs:
        j = xi[xv]
        obs = r & (x == xv)
        for zv in Zs:
            i = zi[zv]
            row = np.zeros(nq + ns)
            for yv in (0, 1):
                m = obs & (y == yv)
                denom = m.sum()
                # P(Z=z | Y=y, X=x, R=1); if no observed cases, fall back to the X-marginal
                pz = (z[m] == zv).mean() if denom else ((z[obs] == zv).mean() if obs.sum() else 1.0 / nz)
                row[yv * nx + j] = pz
            k = nq + 2 * (j * nz + i)
            row[k], row[k + 1] = 1.0, -1.0
            A_eq.append(row)
            b_eq.append(np.mean((~r) & (x == xv) & (z == zv)))
        # q[0,x] + q[1,x] = P(X=x, R=0)
        row = np.zeros(nq + ns)
        row[j] = row[nx + j] = 1.0
        A_eq.append(row)
        b_eq.append(np.mean((~r) & (x == xv)))
    A_eq, b_eq = np.array(A_eq), np.array(b_eq)
    bounds = [(0, None)] * (nq + ns)

    # Step 1: minimum total slack.
    c_slack = np.r_[np.zeros(nq), np.ones(ns)]
    s1 = linprog(c_slack, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not s1.success:
        return worst_case_bounds(y, r) + ({"status": "lp-failed"},)
    tol = s1.fun + tol_scale / np.sqrt(n)

    # Step 2: min / max E[Y] with slack <= min + tol.
    base = np.mean(r & (y == 1))
    c_obj = np.r_[np.zeros(nx), np.ones(nx), np.zeros(ns)]          # sum_x q[1,x]
    A_ub, b_ub = c_slack[None, :], [tol]
    lo = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    hi = linprog(-c_obj, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return base + lo.fun, base - hi.fun, {"status": "ok", "min_slack": float(s1.fun), "tol": float(tol)}


def shadow_bounds_ci(y, r, z, x=None, n_boot: int = 200, alpha: float = 0.10, tol_scale: float = 0.05,
                     seed: int = 0):
    """Bounds plus a subsampling confidence interval (lower-bound alpha/2 pct, upper-bound 1-alpha/2 pct).

    Keep X coarse (default: one bucket). With many X cells, Z is only weakly informative within each
    cell and the LP solutions drift (tested: bias -0.11 with 3 Elo bins vs +0.002 with one bucket).
    """
    y, r, z = np.asarray(y), np.asarray(r, bool), np.asarray(z)
    x = np.zeros(len(y), int) if x is None else np.asarray(x)
    lo, hi, info = shadow_bounds(y, r, z, x, tol_scale)
    rng = np.random.default_rng(seed)
    los, his = [], []
    n = len(y)
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        a, b, _ = shadow_bounds(y[i], r[i], z[i], x[i], tol_scale)
        los.append(a)
        his.append(b)
    wl, wh = worst_case_bounds(y, r)  # the identified set can never be wider than the no-assumption bounds
    return {"lower": lo, "upper": hi,
            "ci_lower": float(max(np.quantile(los, alpha / 2), wl)),
            "ci_upper": float(min(np.quantile(his, 1 - alpha / 2), wh)), **info}


def bin_quantiles(v, k):
    """Integer bins by quantile (for Z = predicted probability, X = Elo difference)."""
    qs = np.quantile(v, np.linspace(0, 1, k + 1)[1:-1])
    return np.searchsorted(qs, v)

"""Dynamic + stable team strength in ONE logistic regression (Track A, step 2).

Following Guan & Tung (2026), pre-game paired-comparison model of pro LoL map outcomes:

  logit P(blue wins) = side + first_pick * fp
                     + a_blue * (blue team's blue-side form - 0.5) + a_red * (red team's red-side form - 0.5)
                     + (league(blue) + team(blue)) - (league(red) + team(red))

- Dynamic form: exponentially weighted moving average of each team's results, kept separately for games
  on blue side and on red side, updated after every game (pre-game values only). Half-life in games.
- Stable strength: one effect per team, ridge-penalised towards 0, i.e. shrunk towards its LEAGUE's mean
  (an approximation of the MAP estimate of a logistic mixed model). League effects (home league, from the
  two years before) are lightly penalised and identified by cross-league games.
- First pick: a separate term only from 2026 (Riot's "First Selection": the team with selection picks side OR
  draft order), +1 if blue picks first, -1 if red does, 0 before 2026 (when first pick = blue side exactly
  and the side term covers it). It cannot be estimated in the 2024 tuning or 2025 report windows.
- Trained end-to-end on weighted log-loss (older games weigh less: half-life in days), refitted daily,
  warm-started from the previous day's coefficients. No post-hoc calibration.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import minimize

from .base import WinModel, day_of

DENSE = ["side", "first_pick", "form_blue_side", "form_red_side"]


def side_form(games: pd.DataFrame, half_life_games: float, prior: float = 0.5):
    """Pre-game blue-side form of the blue team and red-side form of the red team (EWMA of wins)."""
    a = 1 - 0.5 ** (1 / half_life_games)
    fb, fr = {}, {}
    out_b, out_r = np.empty(len(games)), np.empty(len(games))
    bt, rt = games["Blue_teamname"].to_numpy(), games["Red_teamname"].to_numpy()
    y = games["blue_win"].to_numpy(dtype=float)
    for i in range(len(games)):
        b, r = bt[i], rt[i]
        out_b[i], out_r[i] = fb.get(b, prior), fr.get(r, prior)
        fb[b] = out_b[i] + a * (y[i] - out_b[i])
        fr[r] = out_r[i] + a * ((1 - y[i]) - out_r[i])
    return out_b - prior, out_r - prior


class StrengthModel(WinModel):
    name = "strength"

    def __init__(self, form_half_life: float = 10.0, weight_half_life_days: float = 365.0,
                 lam_team: float = 3.0, lam_league: float = 0.1, lam_dense: float = 1e-3,
                 window_days: float = 1460.0, **params):
        super().__init__(form_half_life=form_half_life, weight_half_life_days=weight_half_life_days,
                         lam_team=lam_team, lam_league=lam_league, lam_dense=lam_dense,
                         window_days=window_days, **params)
        self.coef_ = None

    @property
    def features(self):
        form = [f"blue team's blue-side form, red team's red-side form (EWMA, half-life {self.params['form_half_life']} games)"] if self.use_form else []
        return ["side (intercept)", "first pick (+1 blue / -1 red, 2026 only)"] + form + [
                "team effects shrunk to league mean (ridge)", "league effects (home league, cross-league games)"]

    @property
    def caveats(self):
        return ["First pick is a separate term only from 2026 (Riot's First Selection rule). Before 2026 first pick "
                "was always blue side, so it is folded into the side term and CANNOT be estimated in the 2024 tuning "
                "window or in 2025; any first-pick coefficient comes from 2026 games only."]

    def fitted(self):
        if not self.coef_:
            return {}
        return {k: round(float(self.coef_[k]), 4) for k in self.dense_names if k in self.coef_}

    # ------------------------------------------------------------------ design
    use_form = True

    @property
    def dense_names(self):
        return DENSE if self.use_form else DENSE[:2]

    def _dense(self, games):
        if self.use_form:
            fb, fr = side_form(games, self.params["form_half_life"])
        fp = games["blue_first_pick"].to_numpy(dtype=float) if "blue_first_pick" in games else np.ones(len(games))
        # Riot's 2026 "First Selection" rule: the team with selection picks EITHER side OR draft order. Before
        # 2026 first pick = blue side exactly, so the side term covers both and the first-pick term is 0 there;
        # from 2026 it is +1 (blue picks first) / -1 (red picks first). Not identifiable in pre-2026 windows.
        from2026 = (games["date"].dt.year >= 2026).to_numpy()
        cols = [np.ones(len(games)), np.where(from2026, 2 * fp - 1, 0.0)]
        if self.use_form:
            cols += [fb, fr]
        return np.column_stack(cols)

    def walk_forward(self, games, predict_idx):
        from ..league import HOME_WINDOW_DAYS, home_leagues
        p = self.params
        y = games["blue_win"].to_numpy(dtype=float)
        dates = games["date"]
        days = day_of(dates)
        D = self._dense(games)
        bt, rt = games["Blue_teamname"].to_numpy(), games["Red_teamname"].to_numpy()
        out = np.full(len(predict_idx), 0.5)
        order = np.argsort(days[predict_idx], kind="stable")
        state = {}            # warm start: coefficient by name
        cur_day, model, home, home_month = None, None, {}, None
        for j in order:
            i = predict_idx[j]
            if days[i] != cur_day:
                cur_day = days[i]
                month = pd.Timestamp(cur_day).to_period("M")
                if month != home_month:      # home leagues from the two years before, refreshed monthly
                    start = pd.Timestamp(cur_day) - pd.Timedelta(days=HOME_WINDOW_DAYS)
                    past = ((dates < pd.Timestamp(cur_day)) & (dates >= start)).to_numpy()
                    home, home_month = home_leagues(games[past]), month
                model = self._fit_day(cur_day, dates, days, D, y, bt, rt, home, state)
            if model is not None:
                out[j] = self._predict_one(model, D[i], bt[i], rt[i], home)
        self.coef_ = state
        scale = self.params.get("logit_scale", 1.0)
        if scale != 1.0:     # temperature scaling, fitted on 2024 tuning predictions only (scripts/tune_strength.py)
            z = np.log(out / (1 - out))
            out = 1 / (1 + np.exp(-scale * z))
        return out

    # ------------------------------------------------------------------ fit
    def _fit_day(self, day, dates, days, D, y, bt, rt, home, state):
        p = self.params
        age = (np.datetime64(day) - days) / np.timedelta64(1, "D")
        train = (age > 0) & (age <= p["window_days"])
        if train.sum() < 200:
            return None
        w = np.power(0.5, age[train] / p["weight_half_life_days"])
        Dt, yt, b, r = D[train], y[train], bt[train], rt[train]
        teams = sorted(set(b) | set(r))
        tix = {t: k for k, t in enumerate(teams)}
        leagues = sorted({home.get(t, "?") for t in teams})
        lix = {l: k for k, l in enumerate(leagues)}
        n, nt, nl, nd = len(yt), len(teams), len(leagues), D.shape[1]
        rows = np.r_[np.arange(n), np.arange(n)]
        T = sp.csr_matrix((np.r_[np.ones(n), -np.ones(n)], (rows, np.r_[[tix[x] for x in b], [tix[x] for x in r]])), shape=(n, nt))
        Lm = sp.csr_matrix((np.r_[np.ones(n), -np.ones(n)],
                            (rows, np.r_[[lix[home.get(x, "?")] for x in b], [lix[home.get(x, "?")] for x in r]])), shape=(n, nl))
        X = sp.hstack([sp.csr_matrix(Dt), T, Lm]).tocsr()
        lam = np.r_[0.0, np.full(nd - 1, p["lam_dense"]), np.full(nt, p["lam_team"]), np.full(nl, p["lam_league"])]
        names = list(self.dense_names) + [("t", t) for t in teams] + [("l", l) for l in leagues]
        x0 = np.array([state.get(nm, 0.0) for nm in names])
        wsum = w.sum()

        def f(beta):
            z = X @ beta
            loss = np.sum(w * (np.logaddexp(0, z) - yt * z)) / wsum
            g = X.T @ (w * (1 / (1 + np.exp(-z)) - yt)) / wsum
            reg = 0.5 * np.sum(lam * beta ** 2) / n * 100     # penalty scaled per 100 games
            return loss + reg, g + lam * beta / n * 100

        res = minimize(f, x0, jac=True, method="L-BFGS-B", options={"maxiter": 200})
        beta = res.x
        state.clear()
        state.update(zip(names, beta))
        return {"dense": beta[:nd], "team": {t: beta[nd + k] for t, k in tix.items()},
                "league": {l: beta[nd + nt + k] for l, k in lix.items()}}

    @staticmethod
    def _predict_one(model, d, b, r, home):
        z = float(d @ model["dense"])
        z += model["team"].get(b, 0.0) - model["team"].get(r, 0.0)
        z += model["league"].get(home.get(b, "?"), 0.0) - model["league"].get(home.get(r, "?"), 0.0)
        return 1 / (1 + np.exp(-z))


class StrengthNoForm(StrengthModel):
    """Same model without the dynamic side-form terms: stable team + league strength, side, first pick only."""
    name = "strength_noform"
    use_form = False


def margin_rating(games: pd.DataFrame, col, k: float) -> np.ndarray:
    """Pre-game margin-rating difference (blue minus red), an online least-squares Elo on a game margin.

    Each team has one rating on the margin's scale. Before game i the expected blue margin is
    R_blue - R_red + h (h = running side advantage). After the game, if the margin was recorded:
        e = margin - expected;  R_blue += k*e;  R_red -= k*e;  h += 0.01*e
    Games without a recorded margin leave ratings unchanged. Only values before game i are ever returned.
    """
    if isinstance(col, np.ndarray):
        m = col
    else:
        m = games[col].to_numpy(dtype=float) if col in games else np.full(len(games), np.nan)
    bt, rt = games["Blue_teamname"].to_numpy(), games["Red_teamname"].to_numpy()
    R, h = {}, 0.0
    out = np.empty(len(games))
    for i in range(len(games)):
        b, r = bt[i], rt[i]
        rb, rr = R.get(b, 0.0), R.get(r, 0.0)
        out[i] = rb - rr
        if not np.isnan(m[i]):
            e = m[i] - (rb - rr + h)
            R[b], R[r], h = rb + k * e, rr - k * e, h + 0.01 * e
    return out


class StrengthMargin(StrengthNoForm):
    """strength_noform + one margin-rating difference as an extra dense term (Track A, step 3)."""
    margin_col = None
    margin_label = ""

    def __init__(self, margin_k: float = 0.05, **params):
        super().__init__(margin_k=margin_k, **params)

    @property
    def dense_names(self):
        return DENSE[:2] + ["margin_rating"]

    @property
    def features(self):
        f = super().features
        return f[:2] + [f"{self.margin_label} rating difference (online, k={self.params['margin_k']}, past games only)"] + f[2:]

    def _dense(self, games):
        base = super()._dense(games)
        return np.column_stack([base, margin_rating(games, self.margin_col, self.params["margin_k"])])


class StrengthGold15(StrengthMargin):
    name, margin_col, margin_label = "strength_gold15", "m_gold15", "gold-lead-at-15 (k gold)"


class StrengthGold25(StrengthMargin):
    name, margin_col, margin_label = "strength_gold25", "m_gold25", "gold-lead-at-25 (k gold)"


class StrengthLength(StrengthMargin):
    name, margin_col, margin_label = "strength_length", "m_length", "win-quickness (game length)"


WL_PSEUDO_MARGIN = 2.42   # mean |gold lead at 15| in thousands, measured on 2022-2023 only (before any tuning window)


def tiered_gold15(games: pd.DataFrame, w_major: float) -> np.ndarray:
    """Margin used to update the gold-15 rating: a blend of the real gold lead and a W/L pseudo-margin.

    margin = w * gold15 + (1 - w) * (+/-WL_PSEUDO_MARGIN for a blue win/loss), with
    w = 1 for ERL and Other games, w = w_major for Major games (0 = W/L only), and w = 0 when the game has
    no gold data (falls back to W/L). Every game therefore updates the rating.
    """
    g15 = games["m_gold15"].to_numpy(dtype=float) if "m_gold15" in games else np.full(len(games), np.nan)
    tier = games["tier"].to_numpy() if "tier" in games else np.full(len(games), "other")
    bw = games["blue_win"].to_numpy(dtype=float)
    wl = np.where(bw == 1, WL_PSEUDO_MARGIN, -WL_PSEUDO_MARGIN)
    w = np.where(tier == "major", w_major, 1.0)
    w = np.where(np.isnan(g15), 0.0, w)
    out = w * np.nan_to_num(g15) + (1 - w) * wl
    # A game without a result (e.g. a future game added to get a forecast) must not move anyone's rating.
    return np.where(np.isnan(bw), np.nan, out)


class StrengthGold15Tier(StrengthMargin):
    """Gold-15 margin rating with a tier-specific margin weight and W/L fallback (Track A, step 3 revision)."""
    name, margin_label = "strength_g15_tier", "tier-weighted gold-lead-at-15 (k gold; W/L where no gold data)"

    def __init__(self, margin_k: float = 0.16, w_major: float = 0.0, **params):
        super().__init__(margin_k=margin_k, w_major=w_major, **params)

    @property
    def features(self):
        f = super().features
        f[2] += f"; margin weight majors {self.params['w_major']}, ERL/Other 1"
        if self.params.get("logit_scale", 1.0) != 1.0:
            f.append(f"temperature scaling: logit x {self.params['logit_scale']:.3f} (fitted on 2024 only)")
        return f

    def _dense(self, games):
        base = StrengthNoForm._dense(self, games)
        return np.column_stack([base, margin_rating(games, tiered_gold15(games, self.params["w_major"]),
                                                    self.params["margin_k"])])


class StrengthGold15TierCal(StrengthGold15Tier):
    name = "strength_g15_tier_cal"

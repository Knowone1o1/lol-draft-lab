"""Track A step 1: metrics, series bootstrap, holdout lock, leakage sentinel, end-to-end report."""
import numpy as np
import pandas as pd
import pytest

from draftlab import config
from draftlab.eval import bootstrap as bs
from draftlab.eval import metrics as M
from draftlab.eval.dataset import roster_changes, tier_of
from draftlab.eval.harness import eval_mask, evaluate
from draftlab.winmodels import REGISTRY, get_model


@pytest.fixture(scope="module")
def eval_games(synthetic_games):
    from draftlab.data import add_elo, annotate_series
    g, _ = annotate_series(synthetic_games)
    g, _ = add_elo(g)
    g = g.sort_values("date", kind="stable").reset_index(drop=True)
    g["tier"] = g["league"].map(tier_of)
    rb, rr = roster_changes(g)
    g["roster_change_3plus"] = (rb >= 3) | (rr >= 3)
    rng = np.random.default_rng(1)          # synthetic post-game margins correlated with the result
    sign = np.where(g["blue_win"] == 1, 1.0, -1.0)
    g["m_gold15"] = sign * rng.gamma(2, 1.5, len(g)) + rng.normal(0, 1, len(g))
    g["m_gold25"] = sign * rng.gamma(2, 3.0, len(g))
    g["m_length"] = sign * rng.uniform(0.25, 2, len(g))
    g.loc[rng.random(len(g)) < 0.1, "m_gold15"] = np.nan
    return g


@pytest.fixture(autouse=True, scope="module")
def synthetic_draft_features(eval_games):
    """Draft models read a pre-built feature table; inject one (correlated with the result) for the synthetic games."""
    from draftlab.winmodels import draft
    rng = np.random.default_rng(2)
    sign = np.where(eval_games["blue_win"] == 1, 1.0, -1.0)
    cols = {}
    for pre in ("", "b1_", "p1_"):
        for c in draft.CHAMP:
            cols[pre + c] = 0.3 * sign + rng.normal(0, 1, len(eval_games))
    for c in draft.COMP:
        cols[c] = 0.2 * sign + rng.normal(0, 1, len(eval_games))
    draft.FEATURES_OVERRIDE = pd.DataFrame(cols, index=eval_games["gameid"].to_numpy())
    yield
    draft.FEATURES_OVERRIDE = None


def test_calibration_metrics_on_simulated_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, 20000)
    y = (rng.random(20000) < p).astype(int)
    slope, _ = M.calibration_slope(y, p)
    assert abs(slope - 1) < 0.08 and M.ece(y, p) < 0.02
    z = np.log(p / (1 - p)) * 2                      # overconfident: logits doubled
    p_over = 1 / (1 + np.exp(-z))
    slope_over, _ = M.calibration_slope(y, p_over)
    assert abs(slope_over - 0.5) < 0.06 and M.ece(y, p_over) > M.ece(y, p)
    assert M.logloss_terms([1, 0], [0.5, 0.5]).mean() == pytest.approx(np.log(2))


def test_series_bootstrap_is_wider_than_game_bootstrap():
    rng = np.random.default_rng(1)
    series = np.repeat(np.arange(300), 3)                       # 3 games per series, strongly correlated
    v = np.repeat(rng.normal(0, 1, 300), 3) + rng.normal(0, 0.1, 900)
    _, lo_s, hi_s = bs.mean_ci(v, series, B=2000)
    _, lo_g, hi_g = bs.mean_ci(v, np.arange(900), B=2000)
    assert (hi_s - lo_s) > 1.4 * (hi_g - lo_g)


def test_holdout_is_locked(eval_games):
    hold = pd.Timestamp(config.HOLDOUT_START)
    normal = eval_mask(eval_games, "2025-06-01", final=False)
    assert normal.any() and not (eval_games.loc[normal, "date"] >= hold).any()
    final = eval_mask(eval_games, "2025-06-01", final=True)
    assert (eval_games.loc[final, "date"] >= hold).all()


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_no_leakage_sentinel(eval_games, name):
    """Flip every result after a cutoff: no prediction at or before the cutoff may change."""
    g = eval_games
    cut = pd.Timestamp("2026-02-01")
    idx = np.nonzero(((g["date"] >= "2025-09-01") & (g["date"] < cut)).to_numpy())[0]
    base = get_model(name).walk_forward(g, idx)
    flipped = g.copy()
    later = flipped["date"] >= cut
    flipped.loc[later, "blue_win"] = 1 - flipped.loc[later, "blue_win"]
    for c in ("m_gold15", "m_gold25", "m_length"):      # post-game margins must not leak either
        flipped.loc[later, c] = -flipped.loc[later, c] * 3
    from draftlab.winmodels import draft
    draft._BASE_CACHE.clear()
    again = get_model(name).walk_forward(flipped, idx)
    assert np.allclose(base, again), f"{name} uses future results"


def test_evaluate_writes_comparable_run(eval_games, tmp_path):
    models = [get_model(n) for n in ("always_blue", "elo", "elo_league")]
    res = evaluate(eval_games, models, start="2025-06-01", B=200, out_dir=tmp_path)
    for suffix in (".md", ".csv", "_meta.json", "_calibration.svg", "_predictions.csv.gz"):
        assert (tmp_path / f"{res.run_id}{suffix}").exists()
    assert (tmp_path / "index.csv").exists() and not (tmp_path / "holdout_log.json").exists()
    t = res.table
    assert set(t["split"]) >= {"all", "major", "erl"}
    elo = t[(t.model == "elo") & (t.split == "all") & (t.metric == "log_loss")].iloc[0]
    blue = t[(t.model == "always_blue") & (t.split == "all") & (t.metric == "log_loss")].iloc[0]
    assert elo.ci_low <= elo.value <= elo.ci_high
    assert elo.value < blue.value                               # synthetic teams differ in strength: Elo helps
    assert res.meta["final_evaluation"] is False and res.meta["eval_window"][1] < config.HOLDOUT_START


def test_draft_offset_learns_from_features(eval_games):
    """The A4 offset model must actually use the draft features (else its leakage sentinel would be vacuous)."""
    from draftlab.winmodels import draft
    g = eval_games
    idx = np.nonzero((g["date"] >= "2025-09-01").to_numpy())[0]
    draft._BASE_CACHE.clear()
    base = get_model("strength_g15_tier").walk_forward(g, idx)
    m = get_model("draft_champ")
    p = m.walk_forward(g, idx)
    assert np.abs(p - base).max() > 0.02
    assert all(v > 0 for v in m.coef_.values()), m.coef_          # synthetic features favour the winner
    y = g["blue_win"].to_numpy()[idx]
    assert M.logloss_terms(y, p).mean() < M.logloss_terms(y, base).mean()

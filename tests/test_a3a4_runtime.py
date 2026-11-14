"""A6: the A3 + A4 runtime win model (synthetic artifact; no build needed)."""
import json

import numpy as np
import pytest

from draftlab.winmodels.runtime import A3A4WinModel


@pytest.fixture
def model(tmp_path):
    art = {"name": "a3a4", "built": "x", "data_to": "x",
           "strength": {"model": "strength_g15_tier_cal", "params": {}, "fit_day": "x", "scale": 0.9, "side": 0.1,
                        "first_pick": 0.05, "margin_weight": 0.3, "score": {"A": 0.4, "B": -0.2},
                        "home": {"A": "LCK", "B": "LPL", "C": "LPL"}, "league_score": {"LCK": 0.3, "LPL": 0.25}},
           "draft": {"model": "draft_champ", "features": list("abcde"), "beta_per_sd": [0.2, 0.1, 0, 0, 0],
                     "mu": [0.01, 0, 0, 0, 0], "sd": [0.1, 0.2, 1, 1, 1], "coef": {}, "fitted_through": "x"}}
    p = tmp_path / "a3a4_win.json"
    p.write_text(json.dumps(art))
    return A3A4WinModel(p)


def test_strength_logit_is_side_pick_and_scores(model):
    assert model.strength_logit("A", "B", True) == pytest.approx(0.9 * (0.1 + 0.05 + 0.6))
    assert model.strength_logit("A", "B", False) == pytest.approx(0.9 * (0.1 - 0.05 + 0.6))
    assert model.strength_logit("A", "A", True) == pytest.approx(0.9 * 0.15)       # same team: side + pick only


def test_unknown_team_gets_league_score(model):
    assert model.team_score("C") == 0.25 and model.team_score("Nobody") == 0.0


def test_game_without_result_does_not_move_ratings():
    """Regression: a future game added for a forecast (blue_win NaN) used to count as a blue loss."""
    import pandas as pd
    from draftlab.winmodels.strength import margin_rating, tiered_gold15
    g = pd.DataFrame({"Blue_teamname": ["A", "A", "B"], "Red_teamname": ["B", "B", "A"],
                      "blue_win": [1.0, np.nan, np.nan], "m_gold15": [np.nan, np.nan, np.nan],
                      "tier": ["major", "major", "major"]})
    m = tiered_gold15(g, 0.25)
    assert np.isnan(m[1]) and np.isnan(m[2])
    r = margin_rating(g, m, 0.16)
    assert r[1] == r[2] * -1 == pytest.approx(0.16 * 2.42 * 2)      # both future rows see the same ratings


def test_draft_layer_and_level(model):
    x = np.array([0.11, 0.2, 0, 0, 0])
    assert model.draft_logit(x) == pytest.approx(0.2 * 1.0 + 0.1 * 1.0)
    z = model.strength_logit("A", "B", True) + 0.3
    assert model.prob("A", "B", True, x) == pytest.approx(1 / (1 + np.exp(-z)))
    assert model.level_prob(np.zeros(5)) == 0.5                                     # identical drafts: exactly 50%
    assert model.level_prob(x) > 0.5

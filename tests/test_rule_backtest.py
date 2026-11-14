"""Early-game rules, Part 3: the fitting, bootstrap and labelling pieces (synthetic data)."""
import numpy as np
import pandas as pd

from draftlab.rules.backtest import label, logit_fit, ols_fit, series_bootstrap
from draftlab.rules.card import evidence_for


def test_logit_with_offset_recovers_effect():
    rng = np.random.default_rng(0)
    n = 40000
    off = rng.normal(0, 1, n)                    # team strength logit, coefficient fixed at 1
    x = rng.integers(0, 2, n).astype(float)
    y = (rng.random(n) < 1 / (1 + np.exp(-(off + 0.3 * x)))).astype(float)
    b = logit_fit(np.column_stack([np.ones(n), x]), y, off)
    assert abs(b[0]) < 0.05 and abs(b[1] - 0.3) < 0.06


def test_ols_and_series_bootstrap():
    rng = np.random.default_rng(1)
    n = 3000
    df = pd.DataFrame({"series_id": np.repeat(np.arange(n // 3), 3), "x": rng.integers(0, 2, n).astype(float)})
    df["y"] = -500 * df["x"] + rng.normal(0, 1500, n)
    fit = lambda d: ols_fit(np.column_stack([np.ones(len(d)), d["x"]]), d["y"].to_numpy())[1]  # noqa: E731
    est, lo, hi = series_bootstrap(fit, df, B=200, seed=0)
    assert lo < est < hi and lo < -500 < hi and hi < 0


def test_labels():
    assert label(0.01, 0.05, +1, 1000) == "Supported"
    assert label(-0.05, -0.01, +1, 1000) == "Contradicted"
    assert label(-0.05, -0.01, -1, 1000) == "Supported"
    assert label(-0.01, 0.05, +1, 1000) == "Unclear"
    assert label(0.01, 0.05, +1, 100) == "Unclear"          # too few games to call


def test_evidence_on_card():
    rule = {"id": "R19", "oe_testable": True}
    assert evidence_for(rule, None) == ("Not tested yet", None)
    lab, detail = evidence_for(rule, {"R19": {"label": "Supported", "effect": "+2.0 pp", "ci": "[+0.5 pp, +3.5 pp]",
                                              "n_games": 1234, "test": "t"}})
    assert lab == "Supported" and "1,234 past games" in detail
    assert evidence_for({"id": "R08", "oe_testable": True}, {"R08": {"label": "Not tested", "why": "no data"}}) == \
        ("Not tested", "no data")

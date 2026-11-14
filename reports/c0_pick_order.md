# C0: pick-order fix (2026 First Selection), before vs after

Validation folds of `scripts/build.py` (the 6 months before the frozen holdout; the holdout is not scored). Before = every game read as blue-picks-first; after = the side that actually picked first (Oracle's Elixir `firstPick`) takes the first-pick sequence, bans included. Top-5 = the champion actually chosen is in the model's top 5. Differences are after minus before with a 95% interval resampling whole games.

| Action | Variant | Games | Before top-5 | After top-5 | Δ [95% CI] |
|---|---|---|---|---|---|
| ban1 (all) | adj/base | 2,722 | 44.3% | 44.1% | -0.2 pp [-0.3, -0.0] |
| ban1 (red picked first) | adj/base | 1,185 | 46.6% | 46.2% | -0.4 pp [-0.8, -0.1] |
| ban1 (blue picked first) | adj/base | 1,537 | 42.5% | 42.5% | +0.0 pp [-0.0, +0.1] |
| ban1 (all) | raw/base | 2,722 | 44.4% | 44.2% | -0.2 pp [-0.3, -0.0] |
| ban1 (red picked first) | raw/base | 1,185 | 46.8% | 46.4% | -0.4 pp [-0.8, -0.1] |
| ban1 (blue picked first) | raw/base | 1,537 | 42.5% | 42.5% | +0.0 pp [+0.0, +0.1] |
| ban1 (all) | raw/split | 2,722 | 44.1% | 43.9% | -0.2 pp [-0.3, -0.1] |
| ban1 (red picked first) | raw/split | 1,185 | 46.4% | 46.0% | -0.4 pp [-0.7, -0.1] |
| ban1 (blue picked first) | raw/split | 1,537 | 42.3% | 42.2% | -0.1 pp [-0.1, -0.0] |
| ban2 (all) | adj/base | 2,722 | 46.9% | 46.9% | +0.0 pp [-0.1, +0.2] |
| ban2 (red picked first) | adj/base | 1,185 | 48.3% | 48.3% | +0.0 pp [-0.3, +0.3] |
| ban2 (blue picked first) | adj/base | 1,537 | 45.8% | 45.8% | +0.0 pp [-0.0, +0.1] |
| ban2 (all) | raw/base | 2,722 | 46.9% | 46.9% | -0.0 pp [-0.2, +0.1] |
| ban2 (red picked first) | raw/base | 1,185 | 48.3% | 48.2% | -0.1 pp [-0.4, +0.2] |
| ban2 (blue picked first) | raw/base | 1,537 | 45.9% | 45.9% | -0.0 pp [-0.1, +0.0] |
| ban2 (all) | raw/split | 2,722 | 47.0% | 47.0% | -0.1 pp [-0.2, +0.1] |
| ban2 (red picked first) | raw/split | 1,185 | 48.5% | 48.4% | -0.1 pp [-0.4, +0.1] |
| ban2 (blue picked first) | raw/split | 1,537 | 45.9% | 45.9% | -0.0 pp [-0.0, +0.0] |
| pick (all) | adj/base | 2,722 | 49.4% | 49.8% | +0.4 pp [+0.2, +0.6] |
| pick (red picked first) | adj/base | 1,185 | 51.4% | 52.3% | +0.9 pp [+0.5, +1.4] |
| pick (blue picked first) | adj/base | 1,537 | 47.8% | 47.8% | -0.0 pp [-0.1, +0.1] |
| pick (all) | raw/base | 2,722 | 49.3% | 49.9% | +0.6 pp [+0.3, +0.8] |
| pick (red picked first) | raw/base | 1,185 | 51.5% | 52.7% | +1.2 pp [+0.8, +1.7] |
| pick (blue picked first) | raw/base | 1,537 | 47.7% | 47.7% | +0.0 pp [-0.0, +0.1] |

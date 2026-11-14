# LoL Draft Lab: backtest report

Generated 2026-11-12 14:42. Data: 53777 games (2022-01-10 to 2026-11-12). Training leagues: LPL, LCK, LCKC, NACL, EM, LEC, LJL, PRM, LFL, PCS, VCS, NLC, CBLOL, LCS, TCL, LVP SL, WLDs, LCP, MSI, EWC, LTA S, LTA N, FST, LTA.

Method: every month, statistics are rebuilt from games played *before* that month (recency-weighted), so no test game ever sees its own result. Models are fit on earlier months and evaluated on the most recent 4 months: 2026-08 to 2026-11.

Weighting: meta stats use a 90-day half-life with patch boost 3 (each game's stats are targeted at its own patch); lane counters and synergies use a 540-day half-life across all seasons.

## 1. Predicting pro draft actions

For each real pick/ban in held-out games, every available champion is ranked. Top-k = how often the champion the team actually chose was in our top k.

- **pick**: 58950 training actions, 16710 test actions, ~151 candidates each; model log-loss 3.491 (uniform guess: 5.015).

- **ban1**: 35335 training actions, 10018 test actions, ~160 candidates each; model log-loss 3.403 (uniform guess: 5.077).

- **ban2**: 23526 training actions, 6672 test actions, ~149 candidates each; model log-loss 3.367 (uniform guess: 5.005).

| Action | Method | Top-1 | Top-5 | Top-10 | Mean rank |
|---|---|---|---|---|---|
| pick | Model (conditional logit) | 12.0% | 40.5% | 59.7% | 13.3 |
| pick | Baseline: meta presence | 6.2% | 23.7% | 37.6% | 22.0 |
| pick | Baseline: fixed-weight heuristic | 4.0% | 13.8% | 22.8% | 37.2 |
| ban1 | Model (conditional logit) | 13.9% | 43.6% | 64.0% | 12.7 |
| ban1 | Baseline: meta presence | 9.6% | 35.6% | 52.0% | 16.9 |
| ban2 | Model (conditional logit) | 11.6% | 41.8% | 63.2% | 11.5 |
| ban2 | Baseline: meta presence | 4.5% | 20.5% | 34.3% | 22.1 |

### What drives pick decisions (standardised weights)

| Feature | Weight |
|---|---|
| role_fit | +1.40 |
| meta_presence | +1.39 |
| player_comfort | +0.34 |
| team_habit | +0.18 |
| champ_strength | +0.15 |
| comfort_x_late | -0.08 |
| flex_x_early | +0.06 |
| lane_counter | +0.04 |
| flex | -0.04 |
| player_share | +0.03 |
| synergy | -0.01 |

### What drives ban1 decisions (standardised weights)

| Feature | Weight |
|---|---|
| ban_rate | +3.02 |
| meta_presence | -2.08 |
| enemy_comfort | +0.46 |
| own_ban_habit | +0.39 |
| enemy_habit | +0.28 |
| champ_strength | +0.25 |
| enemy_role_fit | -0.22 |
| own_comfort | -0.07 |
| counters_own | +0.00 |

### What drives ban2 decisions (standardised weights)

| Feature | Weight |
|---|---|
| enemy_role_fit | +1.47 |
| meta_presence | +1.38 |
| champ_strength | +0.23 |
| enemy_habit | +0.21 |
| enemy_comfort | +0.19 |
| own_ban_habit | +0.15 |
| ban_rate | -0.14 |
| own_comfort | -0.14 |
| counters_own | +0.06 |

## 2. Win probability: does the draft add information beyond team strength?

5896 training games, 1672 test games. Lower log-loss/Brier is better.

| Model | Log-loss | Brier | Accuracy | AUC |
|---|---|---|---|---|
| Always 'blue wins' rate | 0.6869 | 0.2469 | 56.0% | 0.500 |
| Elo only | 0.6466 | 0.2279 | 62.0% | 0.659 |
| Draft only | 0.6774 | 0.2422 | 57.5% | 0.586 |
| Elo + draft | 0.6458 | 0.2274 | 62.2% | 0.663 |

### Win model weights (Elo + draft)

| Feature | Coefficient |
|---|---|
| elo_diff | +1.801 |
| champ_strength | +3.365 |
| player_comfort | +0.315 |
| synergy | +1.747 |
| lane_counter | +1.112 |
| blue_side (intercept) | +0.123 |

Honest reading: team strength (Elo) explains most of the result; the draft features are judged by how much they improve on Elo alone.

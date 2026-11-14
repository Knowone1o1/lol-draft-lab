# LoL Draft Lab: backtest report

Generated 2026-11-12 13:44. Data: 53777 games (2022-01-10 to 2026-11-12). Training leagues: LPL, LCK, LCKC, NACL, EM, LEC, LJL, PRM, LFL, PCS, VCS, NLC, CBLOL, LCS, TCL, LVP SL, WLDs, LCP, MSI, EWC, LTA S, LTA N, FST, LTA.

Method: every month, statistics are rebuilt from games played *before* that month (recency-weighted), so no test game ever sees its own result. Models are fit on earlier months and evaluated on the most recent 4 months: 2026-08 to 2026-11.

## 1. Predicting pro draft actions

For each real pick/ban in held-out games, every available champion is ranked. Top-k = how often the champion the team actually chose was in our top k.

- **pick**: 58950 training actions, 16710 test actions, ~151 candidates each; model log-loss 3.553 (uniform guess: 5.015).

- **ban1**: 35335 training actions, 10018 test actions, ~160 candidates each; model log-loss 3.527 (uniform guess: 5.077).

- **ban2**: 23526 training actions, 6672 test actions, ~149 candidates each; model log-loss 3.430 (uniform guess: 5.005).

| Action | Method | Top-1 | Top-5 | Top-10 | Mean rank |
|---|---|---|---|---|---|
| pick | Model (conditional logit) | 11.4% | 38.4% | 57.2% | 14.1 |
| pick | Baseline: meta presence | 4.7% | 22.4% | 35.7% | 23.2 |
| pick | Baseline: fixed-weight heuristic | 4.3% | 14.6% | 24.1% | 37.3 |
| ban1 | Model (conditional logit) | 11.8% | 39.7% | 60.2% | 13.9 |
| ban1 | Baseline: meta presence | 7.3% | 29.9% | 46.8% | 20.0 |
| ban2 | Model (conditional logit) | 10.9% | 39.6% | 60.7% | 12.1 |
| ban2 | Baseline: meta presence | 3.9% | 18.5% | 31.9% | 23.6 |

### What drives pick decisions (standardised weights)

| Feature | Weight |
|---|---|
| role_fit | +1.35 |
| meta_presence | +1.29 |
| player_comfort | +0.28 |
| team_habit | +0.20 |
| champ_strength | +0.12 |
| comfort_x_late | -0.06 |
| player_share | +0.06 |
| flex_x_early | +0.04 |
| lane_counter | +0.03 |
| flex | -0.02 |
| synergy | +0.02 |

### What drives ban1 decisions (standardised weights)

| Feature | Weight |
|---|---|
| ban_rate | +2.84 |
| meta_presence | -2.02 |
| enemy_comfort | +0.46 |
| own_ban_habit | +0.38 |
| enemy_habit | +0.31 |
| champ_strength | +0.24 |
| enemy_role_fit | -0.20 |
| own_comfort | -0.10 |
| counters_own | +0.00 |

### What drives ban2 decisions (standardised weights)

| Feature | Weight |
|---|---|
| meta_presence | +1.44 |
| enemy_role_fit | +1.41 |
| ban_rate | -0.25 |
| enemy_habit | +0.21 |
| champ_strength | +0.21 |
| enemy_comfort | +0.19 |
| own_comfort | -0.15 |
| own_ban_habit | +0.14 |
| counters_own | +0.07 |

## 2. Win probability: does the draft add information beyond team strength?

5896 training games, 1672 test games. Lower log-loss/Brier is better.

| Model | Log-loss | Brier | Accuracy | AUC |
|---|---|---|---|---|
| Always 'blue wins' rate | 0.6869 | 0.2469 | 56.0% | 0.500 |
| Elo only | 0.6466 | 0.2279 | 62.0% | 0.659 |
| Draft only | 0.6730 | 0.2401 | 57.7% | 0.599 |
| Elo + draft | 0.6446 | 0.2269 | 62.0% | 0.664 |

### Win model weights (Elo + draft)

| Feature | Coefficient |
|---|---|
| elo_diff | +1.795 |
| champ_strength | +3.156 |
| player_comfort | +0.362 |
| synergy | +2.004 |
| lane_counter | +0.926 |
| blue_side (intercept) | +0.123 |

Honest reading: team strength (Elo) explains most of the result; the draft features are judged by how much they improve on Elo alone.

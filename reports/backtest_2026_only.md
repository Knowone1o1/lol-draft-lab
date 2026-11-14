# LoL Draft Lab: backtest report

Generated 2026-11-12 13:16. Data: 9912 games (2026-01-01 to 2026-11-12). Training leagues: LPL, EM, LCK, LCKC, LEC, LJL, PRM, CBLOL, LCP, LCS, NACL, LFL, EWC, TCL, NLC, VCS, WLDs, PCS, MSI, FST.

Method: every month, statistics are rebuilt from games played *before* that month (recency-weighted), so no test game ever sees its own result. Models are fit on earlier months and evaluated on the most recent 4 months: 2026-08 to 2026-11.

## 1. Predicting pro draft actions

For each real pick/ban in held-out games, every available champion is ranked. Top-k = how often the champion the team actually chose was in our top k.

- **pick**: 36920 training actions, 16710 test actions, ~151 candidates each; model log-loss 3.558 (uniform guess: 5.015).

- **ban1**: 22136 training actions, 10018 test actions, ~160 candidates each; model log-loss 3.565 (uniform guess: 5.077).

- **ban2**: 14736 training actions, 6672 test actions, ~149 candidates each; model log-loss 3.425 (uniform guess: 5.005).

| Action | Method | Top-1 | Top-5 | Top-10 | Mean rank |
|---|---|---|---|---|---|
| pick | Model (conditional logit) | 11.6% | 38.3% | 57.2% | 14.1 |
| pick | Baseline: meta presence | 5.3% | 22.0% | 36.5% | 22.9 |
| pick | Baseline: fixed-weight heuristic | 4.0% | 13.1% | 21.9% | 40.4 |
| ban1 | Model (conditional logit) | 11.9% | 39.0% | 58.0% | 14.6 |
| ban1 | Baseline: meta presence | 7.9% | 31.5% | 48.7% | 19.2 |
| ban2 | Model (conditional logit) | 11.0% | 39.9% | 61.7% | 12.1 |
| ban2 | Baseline: meta presence | 3.9% | 18.6% | 33.1% | 23.1 |

### What drives pick decisions (standardised weights)

| Feature | Weight |
|---|---|
| meta_presence | +1.44 |
| role_fit | +1.44 |
| player_comfort | +0.19 |
| team_habit | +0.14 |
| champ_strength | +0.08 |
| flex | -0.08 |
| comfort_x_late | -0.07 |
| flex_x_early | -0.06 |
| player_share | +0.05 |
| lane_counter | +0.03 |
| synergy | +0.02 |

### What drives ban1 decisions (standardised weights)

| Feature | Weight |
|---|---|
| ban_rate | +2.22 |
| meta_presence | -0.85 |
| own_ban_habit | +0.33 |
| enemy_habit | +0.25 |
| enemy_comfort | +0.24 |
| champ_strength | +0.14 |
| own_comfort | -0.08 |
| enemy_role_fit | +0.01 |
| counters_own | +0.00 |

### What drives ban2 decisions (standardised weights)

| Feature | Weight |
|---|---|
| enemy_role_fit | +1.58 |
| meta_presence | +1.39 |
| champ_strength | +0.18 |
| enemy_habit | +0.14 |
| enemy_comfort | +0.13 |
| own_comfort | -0.13 |
| own_ban_habit | +0.10 |
| ban_rate | -0.09 |
| counters_own | +0.06 |

## 2. Win probability: does the draft add information beyond team strength?

3692 training games, 1672 test games. Lower log-loss/Brier is better.

| Model | Log-loss | Brier | Accuracy | AUC |
|---|---|---|---|---|
| Always 'blue wins' rate | 0.6865 | 0.2467 | 56.0% | 0.500 |
| Elo only | 0.6601 | 0.2339 | 61.2% | 0.639 |
| Draft only | 0.6739 | 0.2406 | 57.4% | 0.599 |
| Elo + draft | 0.6570 | 0.2324 | 61.0% | 0.645 |

### Win model weights (Elo + draft)

| Feature | Coefficient |
|---|---|
| elo_diff | +2.521 |
| champ_strength | +2.600 |
| player_comfort | +0.133 |
| synergy | +1.710 |
| lane_counter | +0.434 |
| blue_side (intercept) | +0.123 |

Honest reading: team strength (Elo) explains most of the result; the draft features are judged by how much they improve on Elo alone.

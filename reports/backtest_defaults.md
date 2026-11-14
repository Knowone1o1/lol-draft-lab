# LoL Draft Lab: backtest report

Generated 2026-11-12 16:46. Data: 53777 games (2022-01-10 to 2026-11-12). Training leagues: LPL, LCK, LCKC, NACL, EM, LEC, LJL, PRM, LFL, PCS, VCS, NLC, CBLOL, LCS, TCL, LVP SL, WLDs, LCP, MSI, EWC, LTA S, LTA N, FST, LTA.

Method: every month, statistics are rebuilt from games played *before* that month (recency-weighted), so no test game ever sees its own result. Models are fit on earlier months and evaluated on the most recent 4 months: 2026-08 to 2026-11.

Weighting: meta stats use a 90-day half-life with patch boost 10 (each game's stats are targeted at its own patch); counters and synergies use a 540-day half-life across all seasons; fearless game weights {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.45, 5: 0.35}; shrinkage k_champ=30, k_pair=20, k_lane=15; Elo K=24.

## 1. Predicting pro draft actions

For each real pick/ban in held-out games, every available champion is ranked. Top-k = how often the champion the team actually chose was in our top k.

- **pick**: 58950 training actions, 16710 test actions, ~151 candidates each; model log-loss 3.120 (uniform guess: 5.015).

- **ban1**: 35335 training actions, 10018 test actions, ~160 candidates each; model log-loss 3.342 (uniform guess: 5.077).

- **ban2**: 23526 training actions, 6672 test actions, ~149 candidates each; model log-loss 3.258 (uniform guess: 5.005).

| Action | Method | Top-1 | Top-5 | Top-10 | Mean rank |
|---|---|---|---|---|---|
| pick | Model (conditional logit) | 21.2% | 51.3% | 69.0% | 10.6 |
| pick | Baseline: meta presence | 6.8% | 24.8% | 40.0% | 21.2 |
| pick | Baseline: fixed-weight heuristic | 4.1% | 13.8% | 23.1% | 37.0 |
| ban1 | Model (conditional logit) | 14.2% | 46.2% | 66.4% | 12.1 |
| ban1 | Baseline: meta presence | 11.4% | 39.8% | 59.5% | 14.8 |
| ban2 | Model (conditional logit) | 14.2% | 46.9% | 67.9% | 10.5 |
| ban2 | Baseline: meta presence | 5.4% | 21.5% | 36.5% | 21.3 |

### What drives pick decisions (standardised weights)

| Feature | Weight |
|---|---|
| meta_presence | +1.55 |
| role_fit | +0.98 |
| copick_sum | +0.49 |
| vspick_max | +0.44 |
| player_comfort | +0.28 |
| duo_copick | +0.27 |
| copick_max | +0.19 |
| team_habit | +0.17 |
| champ_strength | +0.07 |
| flex_x_early | +0.06 |
| comfort_x_late | -0.06 |
| synergy | -0.06 |
| team_counter_sum | +0.04 |
| counter_max | +0.03 |
| synergy_max | +0.03 |
| flex | -0.02 |
| player_share | +0.01 |
| duo_synergy | -0.01 |
| team_counter_max | +0.01 |
| lane_counter | -0.00 |

### What drives ban1 decisions (standardised weights)

| Feature | Weight |
|---|---|
| ban_rate | +3.00 |
| meta_presence | -1.85 |
| enemy_comfort | +0.42 |
| own_ban_habit | +0.36 |
| enemy_role_fit | -0.25 |
| enemy_habit | +0.22 |
| champ_strength | +0.14 |
| own_comfort | -0.07 |
| counters_own | +0.00 |
| enemy_copick_max | +0.00 |
| into_us_max | +0.00 |
| enemy_synergy_max | +0.00 |
| beats_us_max | +0.00 |
| enemy_duo_copick | +0.00 |

### What drives ban2 decisions (standardised weights)

| Feature | Weight |
|---|---|
| enemy_role_fit | +1.45 |
| meta_presence | +1.20 |
| into_us_max | +0.34 |
| enemy_copick_max | +0.31 |
| enemy_habit | +0.20 |
| enemy_comfort | +0.16 |
| own_ban_habit | +0.15 |
| ban_rate | +0.12 |
| own_comfort | -0.10 |
| champ_strength | +0.10 |
| enemy_duo_copick | +0.07 |
| beats_us_max | +0.05 |
| counters_own | +0.05 |
| enemy_synergy_max | -0.01 |

## 2. Win probability: does the draft add information beyond team strength?

5896 training games, 1672 test games. Lower log-loss/Brier is better.

| Model | Log-loss | Brier | Accuracy | AUC |
|---|---|---|---|---|
| Always 'blue wins' rate | 0.6869 | 0.2469 | 56.0% | 0.500 |
| Elo only | 0.6466 | 0.2279 | 62.0% | 0.659 |
| Draft only | 0.6803 | 0.2436 | 57.2% | 0.575 |
| Elo + draft | 0.6460 | 0.2275 | 61.9% | 0.663 |

### Win model weights (Elo + draft)

| Feature | Coefficient |
|---|---|
| elo_diff | +1.820 |
| champ_strength | +3.072 |
| player_comfort | +0.264 |
| synergy | +1.915 |
| lane_counter | +1.240 |
| team_matchups | +0.337 |
| blue_side (intercept) | +0.124 |

Honest reading: team strength (Elo) explains most of the result; the draft features are judged by how much they improve on Elo alone.

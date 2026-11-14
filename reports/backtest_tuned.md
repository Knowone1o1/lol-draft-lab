# LoL Draft Lab: backtest report

Generated 2026-11-12 16:58. Data: 53777 games (2022-01-10 to 2026-11-12). Training leagues: LPL, LCK, LCKC, NACL, EM, LEC, LJL, PRM, LFL, PCS, VCS, NLC, CBLOL, LCS, TCL, LVP SL, WLDs, LCP, MSI, EWC, LTA S, LTA N, FST, LTA.

Method: every month, statistics are rebuilt from games played *before* that month (recency-weighted), so no test game ever sees its own result. Models are fit on earlier months and evaluated on the most recent 4 months: 2026-08 to 2026-11.

Weighting: meta stats use a 45-day half-life with patch boost 10 (each game's stats are targeted at its own patch); counters and synergies use a 5000-day half-life across all seasons; fearless game weights {1: 1.0, 2: 0.65, 3: 0.42250000000000004, 4: 0.274625, 5: 0.17850625000000003}; shrinkage k_champ=20, k_pair=80, k_lane=30; Elo K=24.

## 1. Predicting pro draft actions

For each real pick/ban in held-out games, every available champion is ranked. Top-k = how often the champion the team actually chose was in our top k.

- **pick**: 58950 training actions, 16710 test actions, ~151 candidates each; model log-loss 3.144 (uniform guess: 5.015).

- **ban1**: 35335 training actions, 10018 test actions, ~160 candidates each; model log-loss 3.299 (uniform guess: 5.077).

- **ban2**: 23526 training actions, 6672 test actions, ~149 candidates each; model log-loss 3.271 (uniform guess: 5.005).

| Action | Method | Top-1 | Top-5 | Top-10 | Mean rank |
|---|---|---|---|---|---|
| pick | Model (conditional logit) | 20.9% | 50.6% | 68.7% | 10.8 |
| pick | Baseline: meta presence | 7.0% | 25.5% | 40.2% | 21.0 |
| pick | Baseline: fixed-weight heuristic | 4.2% | 13.6% | 22.4% | 38.4 |
| ban1 | Model (conditional logit) | 14.2% | 47.3% | 67.4% | 11.7 |
| ban1 | Baseline: meta presence | 13.6% | 45.0% | 65.7% | 13.5 |
| ban2 | Model (conditional logit) | 13.9% | 46.5% | 68.4% | 10.6 |
| ban2 | Baseline: meta presence | 5.8% | 22.7% | 38.5% | 20.7 |

### What drives pick decisions (standardised weights)

| Feature | Weight |
|---|---|
| meta_presence | +1.64 |
| role_fit | +1.14 |
| vspick_max | +0.37 |
| copick_sum | +0.36 |
| duo_copick | +0.26 |
| player_comfort | +0.22 |
| copick_max | +0.15 |
| team_habit | +0.15 |
| flex_x_early | +0.07 |
| champ_strength | +0.06 |
| comfort_x_late | -0.06 |
| counter_max | +0.05 |
| flex | -0.05 |
| synergy | -0.05 |
| team_counter_sum | +0.04 |
| synergy_max | +0.02 |
| lane_counter | -0.01 |
| duo_synergy | -0.01 |
| team_counter_max | -0.00 |
| player_share | +0.00 |

### What drives ban1 decisions (standardised weights)

| Feature | Weight |
|---|---|
| ban_rate | +2.71 |
| meta_presence | -1.28 |
| own_ban_habit | +0.29 |
| enemy_comfort | +0.29 |
| enemy_role_fit | -0.28 |
| enemy_habit | +0.19 |
| champ_strength | +0.15 |
| own_comfort | -0.05 |
| counters_own | +0.00 |
| enemy_copick_max | +0.00 |
| into_us_max | +0.00 |
| enemy_synergy_max | +0.00 |
| beats_us_max | +0.00 |
| enemy_duo_copick | +0.00 |

### What drives ban2 decisions (standardised weights)

| Feature | Weight |
|---|---|
| enemy_role_fit | +1.49 |
| meta_presence | +1.33 |
| into_us_max | +0.30 |
| enemy_copick_max | +0.25 |
| enemy_habit | +0.17 |
| own_ban_habit | +0.12 |
| champ_strength | +0.11 |
| enemy_comfort | +0.11 |
| own_comfort | -0.08 |
| ban_rate | +0.08 |
| enemy_duo_copick | +0.08 |
| counters_own | +0.05 |
| beats_us_max | +0.03 |
| enemy_synergy_max | -0.02 |

## 2. Win probability: does the draft add information beyond team strength?

5896 training games, 1672 test games. Lower log-loss/Brier is better.

| Model | Log-loss | Brier | Accuracy | AUC |
|---|---|---|---|---|
| Always 'blue wins' rate | 0.6869 | 0.2469 | 56.0% | 0.500 |
| Elo only | 0.6466 | 0.2279 | 62.0% | 0.659 |
| Draft only | 0.6837 | 0.2453 | 56.4% | 0.563 |
| Elo + draft | 0.6471 | 0.2279 | 62.7% | 0.662 |

### Win model weights (Elo + draft)

| Feature | Coefficient |
|---|---|
| elo_diff | +1.839 |
| champ_strength | +3.412 |
| player_comfort | +0.257 |
| synergy | +1.531 |
| lane_counter | +1.370 |
| team_matchups | +0.313 |
| blue_side (intercept) | +0.125 |

Honest reading: team strength (Elo) explains most of the result; the draft features are judged by how much they improve on Elo alone.

# Weight tuning: within-patch hold-out

Generated 2026-11-12 16:29 in 3055s.

**Method.** Patches are drawn at random from 2026. For each patch, 8 scenarios each keep a random share between 0% and 80% of the patch's games known (chosen at random, all leagues) and delete the rest — from 'patch just released' (0% known: every game of the patch deleted) to 'late in the patch' (80% known). Every statistic is rebuilt without the deleted games (and without later patches/dates), and the deleted games' winners are predicted from draft + players + Elo. A logistic model is scored leave-one-patch-out. Parameters are tuned on the *tune* patches and confirmed on separate *validation* patches.

- Tune patches: 16.03, 16.10, 16.09, 16.13, 16.02, 16.20, 16.07, 16.17, 16.06, 16.15, 16.05, 16.01 (19422 games)
- Validation patches: 16.04, 16.14, 16.11, 16.08, 16.18, 16.16 (8314 games)

## Result

| Params | Tune log-loss | Val log-loss | Val AUC | Val accuracy |
|---|---|---|---|---|
| Start (current) | 0.6337 | 0.6402 | 0.683 | 63.8% |
| Tuned | 0.6322 | 0.6405 | 0.682 | 64.1% |
| Elo only (tuned Elo K) | 0.6469 | 0.6411 | 0.680 | 63.0% |

### By share of the patch known (validation log-loss, lower is better)

| Patch known | Games predicted | Start | Tuned | Elo only |
|---|---|---|---|---|
| 0%-20% | 2373 | 0.6428 | 0.6449 | 0.6432 |
| 20%-40% | 2941 | 0.6392 | 0.6396 | 0.6411 |
| 40%-60% | 1748 | 0.6521 | 0.6497 | 0.6461 |
| 60%-81% | 1252 | 0.6212 | 0.6210 | 0.6299 |

**No reliable improvement on validation patches** — keep current params (gains on tune patches were likely noise).

## Parameters

| Parameter | Start | Tuned | Grid |
|---|---|---|---|
| half_life | 90 | **45** | [30, 45, 60, 90, 135, 200] |
| matchup_half_life | 540 | **5000** | [180, 365, 540, 900, 1500, 2250.0, 3375.0, 5000] |
| patch_boost | 3.0 | **10** | [0, 1, 3, 6, 10, 15.0] |
| fearless_decay | 0.77 | **0.65** | [0.5, 0.65, 0.77, 0.9, 1.0] |
| k_champ | 30.0 | **20** | [10, 20, 30, 50, 80] |
| k_pair | 20.0 | **80** | [5, 10, 20, 40, 80, 120.0] |
| k_lane | 15.0 | **30** | [5, 10, 15, 30, 60] |
| elo_k | 24.0 | **24.0** | [12, 18, 24, 32, 40] |

Fearless game weights (tuned): {1: 1.0, 2: 0.65, 3: 0.42250000000000004, 4: 0.274625, 5: 0.17850625000000003}

## Search path (tune log-loss after each parameter)

- start: 0.6337
- half_life: 0.6330
- matchup_half_life: 0.6329
- patch_boost: 0.6328
- fearless_decay: 0.6327
- k_champ: 0.6326
- k_pair: 0.6324
- k_lane: 0.6324
- elo_k: 0.6324
- half_life: 0.6324
- matchup_half_life: 0.6323
- patch_boost: 0.6322
- fearless_decay: 0.6322
- k_champ: 0.6322
- k_pair: 0.6322
- k_lane: 0.6322
- elo_k: 0.6322
- half_life: 0.6322
- matchup_half_life: 0.6322
- patch_boost: 0.6322
- fearless_decay: 0.6322
- k_champ: 0.6322
- k_pair: 0.6322
- k_lane: 0.6322
- elo_k: 0.6322

## Missing results: binary shadow-variable bounds

Using our model's prediction (5 bins) as the shadow variable Z, Y = blue won, R = result kept. Assumption: Z ⟂ R | Y. The 90% interval comes from subsampling; worst-case assumes nothing.

| Deletion pattern | True blue WR | Naive (kept games) | Shadow 90% CI | Worst case | Width saved |
|---|---|---|---|---|---|
| Random 30% deleted (MAR) | 0.545 | 0.545 | [0.510, 0.563] | [0.386, 0.677] | 82% |
| Blue wins deleted 50%, losses 15% (MNAR) | 0.545 | 0.410 | [0.520, 0.591] | [0.270, 0.611] | 79% |
| Upsets deleted 60% (MNAR) | 0.545 | 0.577 | [0.537, 0.587] | [0.416, 0.695] | 82% |

Note: the 'upsets deleted' pattern depends on the model's own prediction, which breaks the Z ⟂ R | Y assumption on purpose — it shows what happens when the assumption fails.
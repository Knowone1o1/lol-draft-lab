# Weight tuning: within-patch hold-out

Generated 2026-11-12 15:37 in 1307s.

**Method.** Patches are drawn at random from 2026. For each patch, 5 scenarios each keep a random share between 0% and 80% of the patch's games known (chosen at random, all leagues) and delete the rest — from 'patch just released' (0% known: every game of the patch deleted) to 'late in the patch' (80% known). Every statistic is rebuilt without the deleted games (and without later patches/dates), and the deleted games' winners are predicted from draft + players + Elo. A logistic model is scored leave-one-patch-out. Parameters are tuned on the *tune* patches and confirmed on separate *validation* patches.

- Tune patches: 16.04, 16.16, 16.07, 16.09, 16.02, 16.11, 16.01, 16.08 (8055 games)
- Validation patches: 16.05, 16.18, 16.17, 16.20, 16.15, 16.03 (5454 games)

## Result

| Params | Tune log-loss | Val log-loss | Val AUC | Val accuracy |
|---|---|---|---|---|
| Start (current) | 0.6304 | 0.6401 | 0.677 | 63.4% |
| Tuned | 0.6260 | 0.6411 | 0.675 | 63.1% |
| Elo only (tuned Elo K) | 0.6362 | 0.6514 | 0.654 | 61.8% |

### By share of the patch known (validation log-loss, lower is better)

| Patch known | Games predicted | Start | Tuned | Elo only |
|---|---|---|---|---|
| 0%-20% | 1634 | 0.6467 | 0.6442 | 0.6420 |
| 20%-40% | 2085 | 0.6459 | 0.6472 | 0.6609 |
| 40%-60% | 1042 | 0.6281 | 0.6308 | 0.6485 |
| 60%-81% | 693 | 0.6250 | 0.6308 | 0.6488 |

**No reliable improvement on validation patches** — keep current params (gains on tune patches were likely noise).

## Parameters

| Parameter | Start | Tuned | Grid |
|---|---|---|---|
| half_life | 90 | **675.0** | [30, 45, 60, 90, 135, 200, 300.0, 450.0, 675.0] |
| matchup_half_life | 540 | **900** | [180, 365, 540, 900, 1500] |
| patch_boost | 3.0 | **6** | [0, 1, 3, 6, 10] |
| fearless_decay | 0.77 | **0.9** | [0.5, 0.65, 0.77, 0.9, 1.0] |
| k_champ | 30.0 | **30.0** | [10, 20, 30, 50, 80] |
| k_pair | 20.0 | **40** | [5, 10, 20, 40, 80] |
| k_lane | 15.0 | **15.0** | [5, 10, 15, 30, 60] |
| elo_k | 24.0 | **32** | [12, 18, 24, 32, 40] |

Fearless game weights (tuned): {1: 1.0, 2: 0.9, 3: 0.81, 4: 0.7290000000000001, 5: 0.6561}

## Search path (tune log-loss after each parameter)

- start: 0.6304
- half_life: 0.6264
- matchup_half_life: 0.6264
- patch_boost: 0.6263
- fearless_decay: 0.6263
- k_champ: 0.6263
- k_pair: 0.6262
- k_lane: 0.6262
- elo_k: 0.6260
- half_life: 0.6260
- matchup_half_life: 0.6260
- patch_boost: 0.6260
- fearless_decay: 0.6260
- k_champ: 0.6260
- k_pair: 0.6260
- k_lane: 0.6260
- elo_k: 0.6260

## Missing results: binary shadow-variable bounds

Using our model's prediction (5 bins) as the shadow variable Z, Y = blue won, R = result kept. Assumption: Z ⟂ R | Y. The 90% interval comes from subsampling; worst-case assumes nothing.

| Deletion pattern | True blue WR | Naive (kept games) | Shadow 90% CI | Worst case | Width saved |
|---|---|---|---|---|---|
| Random 30% deleted (MAR) | 0.546 | 0.537 | [0.531, 0.598] | [0.376, 0.676] | 78% |
| Blue wins deleted 50%, losses 15% (MNAR) | 0.546 | 0.414 | [0.520, 0.612] | [0.268, 0.621] | 74% |
| Upsets deleted 60% (MNAR) | 0.546 | 0.586 | [0.563, 0.616] | [0.416, 0.706] | 81% |

Note: the 'upsets deleted' pattern depends on the model's own prediction, which breaks the Z ⟂ R | Y assumption on purpose — it shows what happens when the assumption fails.
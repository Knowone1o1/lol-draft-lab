# Weight tuning: summary (2026-11-12)

**Verdict: keep the default weights.** Two tuning runs improved their own tuning patches but not the
held-out validation patches, and the full backtest agrees. No `params.json` was written;
`models/draftlab.pkl` is the default build.

## The two tuning runs (within-patch hold-out, 2026 patches)

| Run | Patches (tune / val) | Scenarios per patch | Tune log-loss start → tuned | Val log-loss start → tuned | Val accuracy start → tuned | Validation gain |
|---|---|---|---|---|---|---|
| 1 | 8 / 6 | 5 | 0.6304 → 0.6260 | 0.6401 → 0.6411 | 63.4% → 63.1% | −0.0010 (worse) |
| 2 | 12 / 6 (different seed) | 8 | 0.6337 → 0.6322 | 0.6402 → 0.6405 | 63.8% → 64.1% | −0.0002 (worse) |

The bar for applying was a validation log-loss gain above 0.0005. Neither run reached it.

The two runs also disagree on the weights themselves, a clear sign the tuning-patch gains were noise:

| Weight | Default | Run 1 | Run 2 |
|---|---|---|---|
| half_life (meta memory, days) | 90 | 675 | 45 |
| matchup_half_life (days) | 540 | 900 | 5000 |
| patch_boost | 3 → now 10* | 6 | 10 |
| fearless_decay | 0.77 (config weights) | 0.9 | 0.65 |
| k_champ | 30 | 30 | 20 |
| k_pair | 20 | 40 | 80 |
| k_lane | 15 | 15 | 30 |
| elo_k | 24 | 32 | 24 |

\* `patch_boost` was raised to 10 in `config.py` in a separate session, based on a draft backtest (ban top-5 +2.4 pts
phase 1, +1.2 phase 2). Both runs leaned the same way. The tuner only scores win prediction, where the boost
barely matters, so it can't judge this weight.

## Full build backtest: defaults vs run-2 weights (Aug–Nov 2026 held out)

Both builds use the current features (including bot-lane duo features) and patch_boost 10.

| Metric | Defaults | Run-2 weights |
|---|---|---|
| Pick top-5 | **51.3%** | 50.6% |
| Ban phase 1 top-5 | 46.2% | **47.3%** |
| Ban phase 2 top-5 | **46.9%** | 46.5% |
| Win log-loss (Elo + draft) | **0.6460** | 0.6471 |
| Win AUC | **0.663** | 0.662 |
| Win accuracy | 61.9% | 62.7% |
| Elo only (same in both) | 0.6466 / 62.0% | 0.6466 / 62.0% |

Mixed and small. Log-loss, the metric the models are trained and tuned on, favours the defaults. The 0.8-point
accuracy difference is within noise (about ±1.2 pts on 1,672 games). Full reports: `backtest_defaults.md`,
`backtest_tuned.md`.

## What it means

- These weights are on a flat plateau: changing them moves win log-loss by ~0.001, about the size of the noise.
  The defaults are as good as this data can show.
- The draft adds only ~0.0006 log-loss over Elo alone in this backtest. Better accuracy will come from new
  information (player-level ratings, form, patch changes, solo-queue practice), not from re-weighting the same stats.

Files: `tune_run1/`, `tune_run2/` (logs, reports, chart data), `tuning_csv/` (run 2 CSVs, shown on the dashboard),
`params_run2_candidate.json` (the rejected weights, kept for reference, not applied).

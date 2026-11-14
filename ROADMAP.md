# Roadmap

Status as of 2026-11-12. Each step stops for review before the next.

## Track A — pre-draft win model (rigorous evaluation)

- [x] **A1 Evaluation harness.** Frozen holdout from 2026-05-12, scored only on "final evaluation". Daily walk-forward. Log-loss and Brier, accuracy and AUC, calibration slope and ECE. Series-level paired bootstrap. Splits: all, major, ERL, other. Per-run reports in `reports/runs/`.
- [x] **A2 Dynamic + stable strength** (Guan & Tung 2026). Side-specific exponentially weighted form, team effects shrunk to league mean, league terms, and a side term, all in one logistic regression. The first-pick term only counts from 2026 (Riot's "First Selection" rule) and can't be estimated before 2026. Tuned on 2024 only; reported on 2025 → holdout.
- [x] **A3 Margin-aware ratings.** (run 20261112-191714; margins help ERL + other leagues, not majors; calibration slope drops to ~0.83; step size at grid edge.) Win/loss only vs + gold at 15 vs + gold at 25 vs + game length, from past games only.
- [x] **A3 revision** (run 20261112-194535): tier-weighted gold-15 margin (majors 0.25, W/L fallback), k 0.16, temperature x0.906 fitted on 2024 -> `strength_g15_tier_cal`, log-loss 0.6219, slope 0.92.
- [x] **A4 Draft ablation** (run 20261112-195011) on `strength_g15_tier_cal` as a fixed offset: full-draft champion features -0.0037 [-0.0052, -0.0021] log-loss; comp tags, interactions and First Selection add nothing on top; residual slope 0.94. Original plan:
  - champion features
  - comp tags (tag-sheet timing and style, early score computed from past data, timing mismatch)
  - draft × strength gap
  - **2026-only pre-draft features:** which team had First Selection, and whether it chose side or pick order
  - splits by game number (fearless games 3–5) and by draft stage
- [ ] **A5 Final evaluation**, only on the user's word.
- [x] **A6** The app picks its win model by config (`config.WIN_MODEL`, env `DRAFTLAB_WIN_MODEL`): "a3a4" (default) = A3 strength + A4 draft layer from models/a3a4_win.json (`scripts/build_a3a4.py`, refit on all games up to now; rebuild after new data), "legacy" = the original Elo + draft model; falls back to legacy if the artifact is missing. Win bar, pick "win chance gain" and ban threat use it; brief and early forecasts keep Elo (trained on it). `/api/state` adds `win_prob_blue_level` (draft alone at equal strength), shown under the win bar. Bug fixed on the way: a game without a result (future game added for a forecast) used to count as a blue loss in the gold-15 rating (`tiered_gold15`); only the first A3+A4 BLG-GEN simulation was affected (rerun).

## Track C — product and fixes

- [x] **E1 Early-game rules machine-readable.** `data/rules/early_game_rules.csv` + `rule_conditions.csv` -> `scripts/build_rules.py` -> `data/rules/rules.json` (36 rules: 22 draft, 12 watch-for, 2 scouting; R20 overrides R08). Build stops if a rule's If text changes until it is re-stamped.
- [x] **E2 Plan card.** `POST /api/plan_card` returns a card per team after a full draft: level 1, jungle start/path, lane priority, first objective, fight windows, draft warnings, watch-for, scouting, gold-at-15 break-even and win condition; every line has rule ID, why, plain condition with values, sources, evidence label. Example: reports/plan_card_example.json. Not yet shown in the web page. Most tag-based rules show "needs labels" until the sheet's early columns are filled.
- [x] **E3 Rule tests.** `scripts/rule_tests.py`, 2024-01 to the holdout, walk-forward facts (same code as the card), strength offset, series bootstrap. Lane priority uses per-lane walk-forward percentile cuts (top/bottom 10% of the last 12 months' forecasts; data/rules/prio_cuts.json; latest top 505 / mid 381 / bot 328 / mid-jungle 587 gold). Supported: R19 (Dragon over Grubs +1.9 pp [+1.0, +2.9]), R17 (-2.1 pp [-3.6, -0.8]), R15, R24, R30 (premise / forecast checks). Unclear: R11, R12, R25, R29, R40. None Contradicted. Labels show on the plan card. See reports/rule_tests.md. R17 tightened to "no Strong lane and at least one Weak" (fires for 21% of teams, was 77%).
- [x] **C0 Pick-order fix.** Draft order follows the side that picked first (Oracle's Elixir firstPick; the first-pick team is assumed to also ban first). Validation top-5 picks 49.3% -> 49.9% (+0.6 pp [+0.3, +0.8]); red-first games +1.2 pp [+0.8, +1.7]; first ban phase -0.4 pp [-0.8, -0.1] in red-first games, which questions the ban-order assumption. See reports/c0_pick_order.md.
- [ ] **C1** Connect the app's series mode to the server-side practice series (Phase 1 API).
- [ ] **C2** Remaining judge items: LPL early-game coverage, sample sizes on win-condition cards.
- [ ] **C3** Portfolio packaging: git and a GitHub repo, a results write-up.

## Track B — analyst loop

- [x] **B1** Data layer (DuckDB: series, games, drafts, players, events, vod_sync).
- [x] **B2** VOD sync (timer OCR at intervals, pause-safe) and clip finder.
- [ ] **B3** Game plan → reality check (deviation flags with clips). Needs events: GRID access or scrim timelines.
- [ ] **B4** AI analyst agent: tools only, every claim cited, plus an evaluation set.
- [ ] **B5** Solo queue → pro translation. Needs a Riot API key and the pro account mapping.

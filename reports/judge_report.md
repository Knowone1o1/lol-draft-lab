# LoL Draft Lab — independent judge report

Reviewed 2026-11-12 by an independent reviewer. I made no changes to code, data, models or params; this file is the only output.
Scope: the live app at http://localhost:8000 (first pass on the model built 16:32; re-checked after the final build: `draftlab.pkl` 17:01, served model stamped 16:47, league offsets 16:58, `app.js` 16:56), at desktop (1366 px) and phone (400 px) width. I played a full Bo3 with fearless (T1 vs Gen.G, Blue then Red then Red) and used every panel. I also read the code in `draftlab/`, `scripts/` and `app/`, and ran my own read-only checks (scripts in my scratchpad, not in the repo).
"Verified" = I reproduced it (numbers or screen). "Suspected" = read from the code, not measured.

## 1. Verdict in 5 lines

1. An org analyst would be impressed by how much is built: leakage-aware monthly snapshots, a tuner, early-game models, a 15-minute plan and a working practice app. They would not trust the numbers yet.
2. First attack: the draft adds no measurable win signal over Elo (held-out log-loss gain 0.0006 ± 0.0022), yet the app ranks picks by a "win chance gain" of +3–4% per pick and shows win chances to one decimal.
3. Second attack: the "hard counter" and whole-team matchup numbers are statistically indistinguishable from noise, and they drive ban suggestions for champions the enemy cannot even play (top ban at Blue ban 5 had 0% enemy pick likelihood).
4. Third attack: team strength. League offsets were added during this review and fix the worst cases, but T1's academy still rates above T1, Galions (ERL) above G2 and Karmine Corp, and the offsets are not used in training or the backtest.
5. Also: design choices were tuned on the reported test months, the README overstates several findings, and the champion strength and counter numbers are not adjusted for team strength.

## 2. Model flaws, most severe first

| # | Severity | Flaw | Where | Status |
|---|---|---|---|---|
| M2 | Critical | The draft adds no measurable win signal beyond Elo, but the app ranks and labels picks by draft win-chance deltas | `engine.py:_recommend_pick`, `reports/backtest.md` §2 | Verified |
| M3 | Major | Whole-team matchups and the "hard counter" ban floor are noise; the floor ignores whether the enemy has an open role | `stats.py:vs_team`, `engine.py:_counter_to_ours`, `_recommend_ban` | Verified |
| M1 | Major (was Critical; partly fixed during review) | Team strength: league offsets exist only at app time, the ordering is still wrong, and there is no roster continuity | `draftlab/data.py` (`add_elo`), `scripts/build_league_elo.py`, `engine.py:team_elo` | Verified |
| M4 | Major | Design choices tuned on the reported test window (forking paths) | `config.py` PATCH_BOOST comment, `reports/backtest_*.md`, `scripts/build_brief.py:34,55-63` | Verified |
| M5 | Major | Tuner's "no future information" claim is false in its default mode | `scripts/tune.py:96-103`, README "Tuning the weights" | Verified (code) |
| M6 | Major | LPL is ~85% missing from every early-game, game-plan and win-condition model | `early.py`, `build_brief.py`, `wincon.py` | Verified |
| M7 | Major | Game-plan "objective value" and "game length" are post-treatment associations presented as advice; README numbers don't match the report | `brief.py:objective_value`, `wincon.py:SimilarComps`, README | Verified |
| M8 | Major | Champion strength and "player comfort" are confounded with team strength and popularity | `stats.py:wr`, `features.py:comp_features` | Suspected |
| M9 | Minor | Pick top-k baselines are too weak; ban-model weights are collinear and not interpretable | `build.py` baselines, `models.py:heuristic_pick_scores`, backtest tables | Verified |
| M10 | Minor | Lane and team gold forecasts are separate models that don't add up, with error 2–4× the size of the forecast | `early.py`, `build_early.py` | Verified |
| M11 | Minor | Win-condition "vs role" differences are mostly within noise, and the UI hides sample sizes | `wincon.py:carry_table`, `reports/comp_timing.md` §4 | Verified (arithmetic) |
| M12 | Minor | Stale objective advice (Atakhan) for 2026, contradicting the project's own report | `brief.py:184,187` | Verified |
| M13 | Minor | The win model is only trained on complete drafts but drives every mid-draft number | `training.py:_month_examples`, `engine.py:win_prob` | Verified: a 3v3 check is OK, so the risk is low |

### M2 — Critical: no measurable draft signal, but picks are sold on it (verified)
- **Final backtest** (`reports/tuning_summary.md`, default build, 1,672 test games):
  - log-loss: Elo + draft 0.6460, Elo only 0.6466
  - accuracy: 61.9% vs 62.0%
  - AUC: 0.663 vs 0.659
- **My independent recomputation** (same code and params, same 12+4 months, `comp_features` on monthly snapshots):
  - the same gain of **0.0006**
  - paired standard error 0.0022; bootstrap 95% CI for the gain [−0.0039, +0.0050]
  - the improvement is not distinguishable from zero
- **The project agrees.** Its own `tuning_summary.md` now says "The draft adds only ~0.0006 log-loss over Elo alone", but the app and README still present per-pick win-chance gains as if they were measured.
- **Tuner validation patches** (`reports/tuning.md`, ~8.3k games): 0.6402 for draft + Elo vs 0.6411 for Elo only. In the 40–60%-known band, Elo only is *better* (0.6461 vs 0.6521).
- **What the app shows anyway.**
  - Every pick suggestion carries a "win chance gain" of the same order as the whole effect: Alistar +3.4%, Xayah +4.5%, Kai'Sa +3.3% (game 1, T1 vs Gen.G).
  - Ban cards say "If they get it: −4.7% win chance for you" (Qiyana, at Blue ban 1, with no picks yet).
  - On the test months the draft terms move the win probability by 3.5 points on average (my check), but that movement has not been shown to predict anything.
  - The "Win chance" sort and the blend's win term rank on these numbers.
- **Consequence.** The headline feature, "picks ranked by win-probability gain", rests on an effect the project's own validation cannot detect. A data scientist will ask for the confidence interval within a minute.
- **Fix.**
  - Lead with what is validated: pick/ban imitation (top-5 51.3%, against 41.4% for a fair baseline; see M9).
  - Present the draft win delta as a small, uncertain tilt ("draft edge: slight / none / unclear"), or shrink the draft coefficients by cross-validated shrinkage.
  - Report intervals (paired bootstrap) for every "beats baseline" claim.
  - Pool more test data (rolling-origin over 2024–2026) so the standard error drops below the effect you hope to show.
- **Effort.** About 1 day.

### M3 — Major: counters are noise; the ban floor fires on impossible picks (verified)
- **Whole-team matchups are pure noise** (`reports/matchups.csv`, the project's own z-scores, n = 16,706 pairs):
  - sd(z) = 0.988, against 1.0 expected under pure noise
  - |z| > 2 for 4.1% of pairs, against 4.6% expected by chance
  - no improvement at 100+ games (sd 0.995)
- **Lane matchups carry only a faint signal:** sd(z) 1.055 overall and 1.18 at 100+ games. The real signal sits in large samples; the top-listed counters in `reports/counters.md` (z ≈ 2–3.6 on 21–35 games) are consistent with the extreme values of ~4,800 tests.
  - The counters page and the Counter explorer say "|z| > 2 is unlikely to be noise". With thousands of comparisons that is false.
  - Examples: Jarvan IV into Ambessa 29 games 86%; Zaahen into Nocturne 21 games 81%.
- **The ban floor fires constantly.** In the final snapshot, 10% of whole-team pairs with ≥15 weighted games clear the +4% "hard counter" bar. For those, the code forces threat ≥ pick rate and damage ≥ half the counter value (`engine.py` `_recommend_ban`).
- **It ignores open roles.** `_counter_to_ours` checks *our* champion's role but never whether the enemy can still play the candidate.
- **Live result** (T1 vs Gen.G, game 1, Blue ban 4; Gen.G had already picked Ambessa top, Jhin bot and Nautilus support, so only jungle and mid were open). 4 of the top 8 ban suggestions were roles Gen.G no longer had open:
  - Yunara: "Hard counter to your Lee Sin (+6%)", "Enemy pick likelihood if open: 0%"
  - Kalista: 0%
  - Camille: 0%, "Kiin (Top)" though top was filled
  - At Blue ban 5 the **#1** suggestion was Yunara with 0% threat. Following the tool wasted a ban.
- **The tuner agrees.** It pushed `k_pair` to the grid maximum (80) and the matchup half-life to 5,000 days (`reports/tuning.md`), i.e. "shrink pair effects as much as possible".
- **README.** The headline example ("Locke wins 62% of the matchup (48 games)") has a 95% CI of roughly 48–76% and was picked from thousands of pairs.
- **Fix.**
  - Use empirical-Bayes shrinkage with the prior variance *estimated* from the z distribution. For whole-team pairs that variance is ≈ 0, so drop them or keep them as text only.
  - Require the candidate to fit an open enemy role (`rp[r] > 0.2` for some open r) before any floor applies.
  - Replace "Hard counter" with "Possible counter (n = 29, could be chance)".
  - Control the false-discovery rate on the counters page.
- **Effort.** 0.5–1 day.

### M1 — Major: team strength is still unreliable across leagues (verified; partly fixed during this review)
- **Before the fix** (first look, model built 16:32). One global Elo pool:
  - Galions (EM/LFL) 1980 and Solary 1930 were above Gen.G 1928
  - T1 Esports Academy (LCKC) 1880 was above Bilibili Gaming 1866
  - T1 1782 was level with T1 Esports Academy Rookies (LAS) 1782
- **The fix.** `scripts/build_league_elo.py` (16:57) now fits league offsets on cross-league games, with a prior of 0 for majors and −200 for academies/ERLs. `engine.py:team_elo` adds the offset.
- **Live after the fix** (`/api/setup`):
  - Gen.G 2082, BLG 1957, **T1 Esports Academy (LCKC) 1946 > T1 1937**, HLE 1934
  - **Galions (LFL) 1878 > Karmine Corp 1821 and G2 1817 (LEC)**
  - LTA N (+44) sits above LEC (+33)
- **What is still wrong.**
  - The offsets are applied only in the app. The win, early-game and game-plan models were trained and backtested on raw Elo (`training.py`, `build_early.py`, `build_brief.py`), including international and EMEA Masters games where the raw ratings are not comparable. The app therefore feeds the models a quantity on a different scale from the one they were fitted on, and the backtest never measured the offsets.
  - Within-league Elo is keyed by team name, with no roster continuity and no regression between splits. That is why an academy that farms its own league can sit above its main team.
  - The prior is strong (SD 120 Elo), so most offsets are close to their assumed values rather than learned.
- **Consequence.** Elo is the largest input to:
  - the win model (+1.8 per 400 points)
  - the team gold-at-15 model (+1,719 gold per 400)
  - the break-even gold in the 15-minute plan

  The team dropdown still shows the raw number, so a coach sees "T1 Esports Academy (1946)" above "T1 (1937)".
- **Fix.**
  - Put the league-adjusted rating into the training features and rerun the backtests.
  - Add regression to the league mean between splits and a player-based roster carry-over, and check offsets against a held-out international event.
  - Show a tier instead of the raw number.
- **Effort.** 1 day.

### M4 — Major: test months reused for design choices (verified)
- **Variant reports.** `reports/` holds `backtest_2026_only.md`, `backtest_2025_2026.md`, `backtest_2022_2026_single_memory.md`, `backtest_two_memory.md` and `backtest_patch.md`. All are scored on the same 2026-08 to 2026-11 test months, and the chosen design (two memories, 2022–26 data, patch boost) is the one that won there.
- **Patch boost.** `config.py` says `PATCH_BOOST = 10.0` was "chosen by the draft backtest (2026-11-12)", i.e. on the test months.
- **Game-plan model.** `build_brief.py` scores four variants on the last 120 days and ships the best (`MIN_GAIN`, lines 34 and 55–63). It reports that same window's log-loss (0.5116 vs 0.5154) as evidence; the winning variant adds 9 learned style weights.
- **The tuner was right, the builder overrode it.** The tuner, which has a separate validation set, found **no** reliable gain from any of these weightings ("keep current params"). Patch boost was then changed anyway, by looking at the test months.
- **The project admits it.** `tuning_summary.md` now says this openly: patch boost 10 came from a draft backtest. It also compared defaults against run-2 weights on the same Aug–Nov months. Honest, but it is still selection on the test set.
- **Consequence.** The reported test numbers are optimistically biased. The bias is small per decision, but there are many decisions and the gains are tiny.
- **Fix.**
  - Freeze an untouched final holdout (e.g. Worlds 2026 plus the last 6 weeks). Make every choice on rolling-origin validation before it, and score the holdout once.
  - List every variant tried in the report.
- **Effort.** Half a day, plus discipline.

### M5 — Major: the tuner sees within-patch future games (verified from code)
- **Code.** In the default (non-`--chrono`) mode, `Evaluator` builds stats "as of the patch's end minus the deleted games", with the known games drawn at random (`tune.py:96-103`).
  - A deleted early-patch game is therefore predicted with player comfort, champion strength and team habits that include *later* games of the same patch.
  - Games from earlier patches played after the deleted game are also included.
- **The claim.** README: "rebuilt without those games, and without later patches or dates, so nothing from the future leaks in". The code contradicts this.
- **Impact** is limited: the tuner's conclusion was "no improvement", and the results themselves are excluded. But the claim is wrong and an analyst will find it.
- **Fix.** Make `--chrono` the default, or reword the README.

### M6 — Major: LPL is mostly missing from the early-game stack (verified)
- **Coverage.** Only **14.5%** of LPL games in the loaded data have gold-difference-at-15. The early-game forecasts, the 15-minute plan, the win-condition tables and the "who needs a lead" tables are effectively trained without the strongest region's habits.
- **How it's described.** `reports/early_game.md` says "Some LPL games lack 15-minute data", which is an understatement.
- **What the app shows.** It gives lane forecasts for LPL teams anyway, and they rest mostly on other leagues' champion and lane statistics.
- **Fix.** Show coverage per league in the report and UI; flag LPL forecasts as low-data, or source the missing LPL timelines.

### M7 — Major: game-plan advice built on conditional associations; README overstates (verified)
- **Objective value.** `objective_value` reports the win-chance difference for first dragon or first tower *at the same gold at 15* and shows it as "Win-chance gain from each objective near even gold: first dragon +15%, first tower +10%" (game 1).
  - Both objectives are partly outcomes of the same game state; what remains after conditioning on gold also carries team quality and lane state the model doesn't see.
  - This is an association, not a value you can bank by taking the drake. The README converts it to "first dragon is worth about +1,200 gold".
- **Game length.** The "Wants to end early / can play long" verdict conditions on game length, which is a consequence of how the game goes (a collider). The gold-band control removes only part of that bias, and the 1.5-SE rule is applied across several objectives with overlapping neighbour sets.
- **Break-even gold.** README: "an early comp facing a late comp needs about **+1,100 gold** at 15 to be a coin flip". `reports/comp_timing.md` §3 (equal teams) says **+345**, so the README overstates the effect about 3×.
- **Fix.**
  - Label these as "teams that took first dragon at this gold went on to win X% more often (association)" and show an interval.
  - Drop the gold-equivalents unless they are estimated properly.
  - Sync the README with the reports.

### M8 — Major: champion strength and "comfort" are confounded (suspected)
- **Champion strength.** `wr(c)` is a raw recency-weighted win rate. It is not adjusted for who played the champion. Champions favoured by top teams inherit those teams' win rates, and the win model then double-counts team strength (Elo and champ_strength together).
- **Player comfort.** "Player comfort" is `log1p(games on champion)`, a count rather than a performance measure. It mostly encodes "this is a meta pick the player plays a lot".
- **Example.** The ban card "If they get it: −4.7% win chance" for Qiyana at ban 1 is driven by Canyon's ~18 games.
- **Fix.** Estimate champion effects in a logistic model with both teams' ratings as offsets (or fixed effects), and use performance-on-champion relative to the player's baseline for comfort.

### M9 — Minor: weak baselines and uninterpretable ban weights (verified)
- **Baseline.** The "meta presence" pick baseline ignores role availability. It will rank a support first when only mid is open, so the headline gap overstates the model's value.
- **My check** (same 4 test months, 16,710 picks, no training needed):

  | Baseline | Top-1 | Top-5 | Top-10 |
  |---|---|---|---|
  | Presence only (as reported) | 6.8% | 24.8% | 40.0% |
  | Presence × role fit | 11.3% | 37.7% | 56.8% |
  | Presence × role fit + 0.5 × player comfort | 12.5% | 41.4% | 60.1% |
  | Model (final backtest) | 21.2% | 51.3% | 69.0% |

  The model still wins, but by ~9 points top-1 and ~10 points top-5, not the 14 and 27 the report implies.
- **Phase-1 bans.** The model's top-5 is 46.2% against 39.8% for presence alone, a gap of only 6 points.
- **Ban weights.** The ban-1 weights `ban_rate +2.84` and `meta_presence −1.89` are near-collinear (presence = pick + ban rate). The "What drives ban1 decisions" table invites a wrong reading ("bans avoid present champions").
- **Top-k framing.** With ~150 candidates of which ~40 are realistic, top-1/top-5 is fine but should be reported against a role-aware baseline.
- **Fix.** Add a role-aware baseline, and report pick rate and ban rate rather than presence.

### M10 — Minor: early forecasts are weak and inconsistent (verified)
- **Accuracy.** Lane R² is 0.09–0.16 and MAE 630–870 gold, while typical predicted lane leads are 100–400 gold.
- **Inconsistency.** The team model is fitted separately from the lane models, so the numbers don't add up on screen:
  - game 1: the five lanes sum to +263 while "Team gold at 15" shows **+484**
  - phone run: lanes sum to −199 while the team shows −41
- **Fix.** Derive team gold from the lanes (or vice versa), show ±MAE bands, and round to the nearest 100.

### M11 — Minor: win-condition differences are mostly noise
- **What it measures.** "vs role" is the gap between the team's win rate with this lane 500+ ahead and with it 500+ behind, minus the role's average gap. Typical values are +4–11%.
- **Noise level.** For a champion with ~200 games (≈70 ahead, ≈70 behind), the standard error of a gap is ≈ 8 points. Thresh (216 games, +5%) and Irelia (79 games, +11%) are within noise; shrinkage reduces the problem but does not remove it.
- **What the UI shows.** Bars like "Lane ahead 75% / Lane behind 27%", with no n.
- **Fix.** Show n and an interval, and only name a "win condition" when the difference is clear.

### M12 — Minor: stale advice text
- `brief.py:184,187` tells the user to take "Atakhan". `reports/comp_timing.md` itself notes Atakhan is gone in 2026.

### M13 — Minor: the win model is used where it wasn't trained
- **Training.** The win model is trained only on complete 5v5 drafts with true roles (`role_hint=actual_roles`).
- **Use.** The app applies it from ban 1 onward with inferred roles and sums/5, and each pick delta is the difference between two partial drafts.
- **My check.** At 3v3 with inferred roles, held-out log-loss is 0.6438, no worse than full drafts, so the risk is lower than it looks. Still, it is unvalidated per stage.
- **Fix.** Validate per draft stage (bans only, 1v1 … 5v5) and show "draft edge" only from the stage where it is validated.

## 3. Style and usability issues, most important first

| # | Severity | Where | Problem |
|---|---|---|---|
| U1 | Was Critical, **fixed during review** | Draft board, champion grid | Clicking a locked or greyed champion froze the draft; fixed at 16:56 and re-verified |
| U2 | Major | Win bar, mini-map, objective bars, scenario buttons | Blue/red mean both "side" and "good/bad"; the map is not flipped for Red |
| U3 | Major | Whole draft screen | Hierarchy: the turn indicator is small, recommendations sit in a nested scroll box, and insights are below the fold |
| U4 | Major | Recommendation cards | Walls of 6–9 tiny grey bullets, with duplicates and jargon |
| U5 | Major | Win bar, pick deltas, lane gold, "Hard counter" | False precision; sample sizes hidden |
| U6 | Major | Setup, team dropdown | Raw Elo shown with absurd ordering |
| U7 | Major | Phone width (400 px) | Setup fields overflow the panel; board is 3,346 px tall; two stacked 640 px scroll traps |
| U8 | Minor | Everywhere | No loading states; error messages were added during the review but only for the draft calls |
| U9 | Minor | Game plan tab | Slider starts at +0 instead of the forecast; axis label clipped; style tags contradict the header |
| U10 | Minor | Counter explorer | Case-sensitive lookup; misleading "z ≥ 2" legend |
| U11 | Minor | Fearless reasons | The lock cost is phrased as a reason to pick |
| U12 | Minor | Keyboard and accessibility | Champion cells and recommendation cards are click-only `div`s; Undo/Back look like filter chips |
| U13 | Minor | Mini-map | Mid lane label sits on top of champion portraits |
| U14 | Minor | Code hygiene | Dead CSS from old layouts; one unexplained HTTP 500 during the session |

**U1 — Was Critical, now fixed: clicking a locked champion froze the draft.**
- **Status.** Fixed in `app.js` at 16:56. I re-tested with a pre-locked Azir: the click shows "Azir is locked for this series (fearless).", `busy` stays false, and the next pick works.
- **Still open.** After a generic API error the attempted action stays in `S.actions`. Roll it back. The original finding follows for the record.
- **Where.** Draft board, champion grid, in game 3 of a fearless series.
- **What.** Locked champions are greyed but still clickable. `choose()` pushes the champion, `/api/state` returns 400 "a champion was used twice", and the exception escapes `refresh()` with `busy = true` (`app.js:167-170, 187-195`).
- **Reproduced.** After clicking Alistar (locked from game 1):
  - `busy` stayed true and the console showed "Uncaught (in promise) Error: a champion was used twice"
  - Undo did nothing, and recommendation clicks were ignored
  - the only exit is "Back to setup", which abandons the series
- **Why it matters.** A coach live in champ select loses the whole session from a single mis-click.
- **Fix.** Ignore clicks on `.off` cells, wrap `refresh()`/`ai_move` in try/finally, roll back the pushed action on error, and show a visible error message.

**U2 — Major: blue/red carry two meanings.**
- **What I saw.**
  - Side colours are blue/red, but the lanes mini-map colours a lane **blue when "you" are ahead** and red when behind, even when you are Red side.
  - Objective bars are blue for "your chance".
  - Elsewhere good/bad is green/red (`.pos/.neg`), the win-condition "neg" is gold, and the "Good case" scenario button has a blue border.
- **Red side, game 2.** T1 was red, Gen.G's portraits had blue rings at the bottom-left base, and T1's +197 mid lead was drawn as a blue lane. A coach reads that as "Gen.G ahead mid".
- **Fix.**
  - Reserve blue/red strictly for sides.
  - Use one good/bad pair everywhere: green/red, or a colour-blind-safe teal/orange.
  - Flip the map so the user's base is always bottom-left.

**U3 — Major: hierarchy is upside down for live use.**
- **Desktop (1366 px).**
  - "Your ban — Blue ban 1" is a 14 px line above a 7-column champion grid that takes ~60% of the board.
  - Recommendations sit in a narrow scroll box.
  - The early-game insights, review and game plan are all below the fold.
- **Draft complete.** The empty grid still dominates the screen and the review needs a scroll.
- **Phone (400 px).** The turn indicator is at y = 727 px, recommendations at 893, the enemy's picks at 2,197 and insights at 2,605 of 3,346 px.
- **Why it matters.** In champ select the coach needs whose turn it is, the 3 best options and the enemy's picks on one screen.
- **Fix.**
  - Make a sticky top bar with the turn, timer slot and win edge.
  - Show the top 3 recommendations as large cards above the grid, and collapse the grid behind search.
  - Put the insights in a right-hand column at ≥1200 px, and use tabs (Board / Suggestions / Plan) on phones.

**U4 — Major: recommendation cards are walls of text.**
- **What.** Each card has 6–9 bullets in 11.5 px muted grey (`.rec ul`).
- **Duplicates.** "Hard counter to your Rumble (+4%): protect your pick" and "Beats your Rumble (+4% win-rate lift, any role)" appear together.
- **Jargon.** "presence", "lift", "2.7x as often as chance", "threat", "Meta: 21% presence, 9% ban rate".
- **Fix.** One bold headline reason, two short chips (e.g. "Chovy 11 games", "counters Rumble?"), and an expand control for the rest. Use coach language ("Chovy plays it a lot", "often picked into your Rumble").

**U5 — Major: false precision and hidden sample sizes.**
- **Decimals.** The win bar shows "37.2% · 62.8%" before any pick, and pick deltas show one decimal ("+3.4%"), for an effect whose 95% CI includes zero (M2).
- **Overstated labels.** Lane gold is shown to the unit ("−189") with an error of ~700. "Hard counter" is stated as fact.
- **Hidden n.** Win-condition bars have no sample size. "First dragon +15%" is shown without caveat.
- **Fix.** Round to whole percentages, add ± ranges or a three-level "edge" label, show n on every rate, and say "could be chance" where it is.

**U6 — Major: raw Elo in the team dropdown.**
- **What.** Entries show a raw rating. Before the fix: "T1 (1782)", "Gen.G (1928)", with Galions at 1980. After the fix: "T1 Esports Academy (1946)" still sits above "T1 (1937)" (M1).
- **Why it matters.** It is the first number a coach sees, and it is visibly wrong.
- **Fix.** Hide it until M1 is fixed, then show a tier.

**U7 — Major: phone layout.**
- **Setup overflows.** At 400 px the setup `select`s and labels extend to the viewport edge, past the panel's right edge (panel 380 px, fields 413 px). The fearless hint ("…rest of the serie") and the "Varied" slider label are clipped.
- **Board.** The board is 3,346 px tall, with two stacked 640 px internal scroll boxes (recommendations, then the grid) that trap the scroll gesture.
- **Game plan chart.** The axis text scales down to ~5 px.
- **Fix.** Use `min-width: 0` / `width: 100%` on the grid2 children and a single scroll on phones; set the chart's font size in CSS pixels, not viewBox units.

**U8 — Minor: no loading or error states.**
- The setup dropdowns are empty for ~5 s on first load with no spinner.
- Sort buttons refetch silently.
- Since 16:56, draft-update and opponent-move errors show a message; the early-game, plan and sort requests still fail silently.

**U9 — Minor: game plan tab.**
- **Slider.** It starts at "+0" even when the forecast is +484, because it is initialised only when the plan first appears at 3v3.
- **Clipping.** The x-axis label "+6,000" is clipped to "+6,00" at desktop width.
- **Contradictions.** The header says "Early · Skirmish", but the style chips say "your Front-to-back vs their Dive". "Style matchup is about neutral" sits next to two "favoured" and two "unfavoured" chips.

**U10 — Minor: Counter explorer.**
- **Case-sensitive lookup.** "azir" returns "Not enough games" instead of Azir's table.
- **Misleading legend.** It says "z ≥ 2 (bold) is unlikely to be noise", which is wrong for thousands of comparisons (M3).

**U11 — Minor: fearless reasons.**
- "Fearless: #1 comfort pick for Oner; locked for up to 1 more game if used now" is listed among the *reasons to pick*. The lock is a cost.
- **Fix.** Show it as a warning chip ("Burns Oner's #1 for the series").

**U12 — Minor: keyboard and accessibility.**
- Champion cells and recommendation cards are `div`s with `onclick`, so they can't be reached or used with the keyboard.
- There are no shortcuts (search on "/", Enter to lock the top recommendation).
- "Undo" and "Back to setup" are styled exactly like the role-filter chips in the same row.
- Fonts of 10.5–11.5 px for reasons and champion names are below comfortable reading size on a coach's laptop.

**U13 — Minor: mini-map label overlap.**
- The mid label ("−188", "−309") is drawn at the map centre on top of the mid and jungle portraits.

**U14 — Minor: code hygiene.**
- `style.css` keeps unused rules from older layouts (`.lane`, `.brief-top`, `.scens`, `.earlybody`).
- One API call returned HTTP 500 during my session; I could not reproduce it.

## 4. What is genuinely solid (verified)

- **No result leakage in the main backtest.** Draft and win features for month *m* come only from games dated before *m* (`training.py:build_examples`, `stats.py` filters `date < ref_date`). Elo is strictly pre-game.
- **Pick imitation beats even a fair, role-aware baseline.** Top-5 is 51.3% against 41.4% for presence × role fit + comfort (my check, 16,710 held-out picks; M9). The edge is real, just smaller than reported.
- **Win probabilities are reasonably calibrated** on held-out months. In my decile check, predicted vs observed ranges from 0.285 vs 0.339 (mild overconfidence on underdogs) to 0.790 vs 0.810.
- **The tuner refused to apply parameters that didn't beat validation.** That is the right discipline, and the report says so plainly.
- **The fearless series flow works end to end**: locks carry over, the loser picks side, and locked champions are never recommended across a Bo3.

## 5. Top 5 fixes, in priority order

1. **Make every claim match the evidence (M2, M4, M5, M7).**
   - Freeze an untouched holdout and report paired bootstrap CIs.
   - Present the draft win effect as "slight / unclear" rather than ±x.x%.
   - Correct the README: +1,100 → +345 break-even, the tuner's "no future" claim, the Locke example's CI, the objective "gold equivalents".
   - *Effort: about 1 day.*
2. **Replace the counter machinery (M3, U10).**
   - Use empirical-Bayes posteriors with an estimated prior variance (drop whole-team pairs, whose variance is ≈ 0) and FDR control on the counters page.
   - Require an open enemy role before any ban floor, and rename "Hard counter" to "Possible counter (n)".
   - *Effort: 0.5–1 day.*
3. **Finish the team-strength fix (M1, U6).**
   - Train and backtest on the league-adjusted rating, add split regression and roster carry-over, and validate the offsets on a held-out international event.
   - Show a tier instead of raw Elo.
   - *Effort: about 1 day.*
4. **Adjust champion and counter stats for team strength (M8).**
   - Fit champion effects with both teams' ratings as offsets, and measure comfort as performance relative to the player's baseline.
   - *Effort: about 1 day.*
5. **Redesign for live use (U2, U3, U4, U5, U7).**
   - One colour language (sides vs good/bad), and flip the map for Red.
   - A sticky turn bar with the top-3 recommendation cards above the grid, and one-line reasons with an expand control.
   - Whole-number percentages with ranges and n, insights beside the board on desktop, and a tabbed phone layout.
   - *Effort: 1–2 days.*

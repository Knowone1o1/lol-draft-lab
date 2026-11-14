# LoL Draft Lab

Draft modelling and practice for professional League of Legends, built on public pro-play data.

- **Draft practice:** draft live against an AI that imitates a specific pro team's pick/ban habits, with your own roster.
- **Player-aware recommendations:** suggestions weigh each player's recent champion pool, not just the meta.
- **Explained suggestions:** every suggestion lists its reasons, e.g. "Faker (Mid): ~12 recent games, 67% WR", "Favoured vs Azir in lane (+3%)" or "Enemy pick likelihood: 18%".
- **Backtested:** models are scored against what pro teams actually drafted in held-out months (`reports/backtest.md`).
- **Fearless-aware:** detects fearless-draft leagues from the data and locks champions used earlier in a series.

## How it works

| Piece | What it does |
|---|---|
| `draftlab/data.py` | Turns Oracle's Elixir CSVs into one row per game: full pick/ban order, players, series grouping, fearless detection and pre-game team Elo. |
| `draftlab/stats.py` | Builds a snapshot with **two memories**. The meta memory (90-day half-life) covers presence, champion strength, player comfort and team habits, so old seasons barely count. The matchup memory (540-day half-life, all seasons) covers lane counters and synergies, measured as lift beyond each champion's strength so an old meta pick doesn't look like a counter. Win rates are shrunk towards sensible priors so small samples don't mislead. |
| `draftlab/matchups.py` | Cross-season counter analysis: for each lane matchup, wins versus the wins expected from both champions' strength *that season*, with a z-score and how many seasons the edge held. Writes `reports/counters.md` and `reports/matchups.csv`, and powers the app's counter explorer. |
| `draftlab/features.py` | Builds features for each candidate champion at each draft step (role fit, player comfort, team habit, meta, synergy, lane counter, flex value, …). |
| `draftlab/models.py` | Holds the **conditional-logit choice models** for picks, phase-1 bans and phase-2 bans ("which champion would this team choose here?"), plus a **logistic win model** (Elo + draft). |
| `draftlab/engine.py` | Recommends picks (by win-probability gain plus roster fit) and bans (by denying the enemy's likely picks), and runs the opponent AI, which samples from the team's imitation model. |
| `scripts/build.py` | Runs the leakage-free monthly backtest, fits the final models and writes `models/draftlab.pkl` and `reports/backtest.md`. |
| `app/` | FastAPI server and a no-build HTML/JS draft board. |

**No look-ahead:** every month's features come from a snapshot of games played *before* that month, so the backtest never sees the result it is predicting.

## Setup (Windows)

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

### Data

1. Open the [Oracle's Elixir downloads page](https://oracleselixir.com/tools/downloads) and follow its Google Drive link.
2. Download `2026_LoL_esports_match_data_from_OraclesElixir.csv` and, ideally, the 2025 file.
3. Put them in `data/raw/`. Any file name works.

If Drive says "quota exceeded", sign in to Google, right-click the file → **Make a copy**, then download the copy from your own Drive.

### Build and run

```bash
.venv\Scripts\python scripts\build.py
.venv\Scripts\python -m uvicorn app.server:app --port 8000
```

Then open http://localhost:8000.

`build.py` options:
- `--leagues LCK LEC LPL`: only train on these leagues.
- `--test-months 4`: how many recent months to hold out for the backtest.
- `--train-months 12`: how many months of drafts to fit on.

### Testing without real data

```bash
.venv\Scripts\python tests\make_synthetic.py %TEMP%\syn\raw
set DRAFTLAB_RAW_DIR=%TEMP%\syn\raw
set DRAFTLAB_MODEL_DIR=%TEMP%\syn\models
set DRAFTLAB_REPORT_DIR=%TEMP%\syn\reports
.venv\Scripts\python scripts\build.py
```

## Evaluating win models (Track A)

`scripts/evaluate.py` scores pre-game win models walk-forward and writes one comparable result set per run to `reports/runs/`: a markdown report, a CSV results table, metadata, a calibration plot and the predictions. Each run also adds a line to `index.csv`.

- **Frozen holdout:** games from `config.HOLDOUT_START` (2026-05-12) on. It is scored only with `--final-evaluation`, and every use is logged.
- **Walk-forward:** every day's games are predicted by models refitted on all earlier days, with pre-game features only. A sentinel test flips every later result and checks that no earlier prediction changes.
- **Metrics:**
  - log-loss and Brier as the headline
  - accuracy and AUC
  - calibration slope and intercept, ECE, and a reliability table and plot
- **Intervals:** 95% bootstrap intervals resample **whole series**, and model-vs-baseline differences are paired.
- **Splits:** all leagues, major leagues (plus Worlds, MSI, First Stand, EWC), ERLs, other.
- **Models** sit behind `draftlab/winmodels` (`WinModel` interface plus registry). Baselines: `always_blue`, `elo`, `elo_league`.

```bash
.venv\Scripts\python scripts\evaluate.py --models always_blue elo elo_league --start 2024-01-01
```

## Data layer (DuckDB)

`scripts/ingest.py` builds `data/draftlab.duckdb`, keyed by `series_id` and `game_id` everywhere.

| Table | Contents |
|---|---|
| `series` | league, teams, format (Bo1/Bo3/Bo5 inferred from the score), fearless flag, score |
| `games` | series, game number, date, patch, sides, result |
| `draft_actions` | all 20 picks and bans in tournament order |
| `game_players` | player and champion per side and role |
| `events` | game_time_s, event_type (kill / objective / ward / tower / item), team, player, x, y |
| `vod_sync`, `vod_sync_points` | video URL and several (game time, video time) points per game, so pauses are handled |

**Sources** sit behind interfaces in `draftlab/sources/`:
- `OraclesElixirSource`: drafts and results.
- `RiotTimelineSource`: events from Riot match-v5 timeline JSON, which is what tournament-code scrims produce.
- `GridSource`: a placeholder until there is GRID access (`GRID_API_KEY`).
- `CsvAccountSource`: pro solo-queue accounts.

**Series state** (`store.series_state`) gives the score, the games played and the champions locked by fearless. `remaining_pool` filters a player's or team's pool against those locks.

**API endpoints:** `GET /api/series`, `GET /api/series/{id}` and `GET /api/games/{id}` read from the store. `POST /api/practice/series` and `POST /api/practice/series/{id}/games` keep practice series in a separate `data/practice.duckdb` and reject champions already locked by fearless.

```bash
.venv\Scripts\python scripts\ingest.py --timelines data/timelines --vod data/vods.csv
```
```bash
.venv\Scripts\python -m pytest tests
```

## VOD sync and clip finder (Phase 2)

`draftlab/vod/` matches recordings to in-game time and turns events into clips.

- **Frames:** read with ffmpeg (the bundled imageio-ffmpeg build), frame-accurate even on multi-hour VODs.
- **Timer reading:**
  - **Template OCR, no installs needed:** it learns the timer font from one known moment. You say "at video 5:12 the timer showed 0:41", and it reads the next 40 seconds to learn every digit.
  - **Tesseract** is used instead if it's installed.
- **Sync at intervals, not one offset:** the timer is read every 30 seconds.
  - A reading only counts if it agrees with the current offset, so misreads are thrown out.
  - When the video keeps running but the clock stops (a pause), two readings in a row must agree on the new offset before it's accepted.
  - The store interpolates between the sync points.
- **`find_clips(query)`:** takes a short text query ("dragons T1 before 15:00", "kills by Faker") or filters, and returns clips with game_id, start, end and reason. Nearby events merge into one clip.
  - Public VODs get a **YouTube `?t=` link** and are never downloaded.
  - Local recordings, such as your scrims, get an **ffmpeg cut**.
  - `clip_at()` gives Phase 3's deviation flags a clip for any moment.
- **Tests:** they include an end-to-end run on a generated video with a 30-second pause: calibrate, sync, check times on both sides of the pause, then cut.

```bash
.venv\Scripts\python scripts\vod_sync.py preview --video scrim.mp4 --at 10:00 --out frame.png
```
```bash
.venv\Scripts\python scripts\vod_sync.py calibrate --video scrim.mp4 --crop 0.47,0.0,0.06,0.04 --video-time 5:12 --game-time 0:41
```
```bash
.venv\Scripts\python scripts\vod_sync.py sync --video scrim.mp4 --crop 0.47,0.0,0.06,0.04 --game-id G1
```
```bash
.venv\Scripts\python scripts\vod_sync.py manual --game-id G1 --url "https://youtu.be/..." --point 0:00=1:02:15 --point 20:00=1:24:40
```
```bash
.venv\Scripts\python scripts\vod_sync.py clips "dragons T1 before 15:00" --cut
```

API: `GET /api/clips?q=...&game_id=...`. Clips need events, which come from scrim timelines (`ingest.py --timelines`) or GRID data.

## League strength (Elo across leagues)

Plain Elo is one pool per league: every league averages 1500, so a dominant ERL team (Galions, 1980) ended up rated above Gen.G (1928). `scripts/build_league_elo.py` fixes this:
- **Home league:** each team gets one.
- **League offsets:** each league gets an offset in Elo points, fitted on about 2,900 cross-league games (internationals, EMEA Masters, KeSPA Cup).
- **Prior:** major leagues start at 0, and academy and ERL leagues at 200 below their parent league.
- **Result:** LCK +154, LPL +91, LEC +33, LCS +20, ERLs −100 to −250. Galions is now 1878 and Gen.G 2082.
- **Scope:** the offset cancels within a league, so it only changes cross-league matchups in the app and the team list, and the trained models are unchanged. LCK Challengers (+66) is estimated mostly from KeSPA Cup games, so treat it with care.

```bash
.venv\Scripts\python scripts\build_league_elo.py
```

## Recommendation order and "this patch" facts

The default order is **Pro priority**: the part of the model that's validated (the pros' actual pick is in its top 5 about 51% of the time). In backtests, the draft adds little to win prediction beyond team strength, so each card shows the win-chance change as a small "win est." with that caveat.

Picks can be sorted three ways:
- **Blend** (default): win-chance gain plus pro priority.
- **Win chance:** what the win model says alone.
- **Pro priority:** what the pick model predicts pros would take here.

The top three pro picks get a *Pro pick #n* badge. `scripts/build_recent.py` writes `models/recent.pkl`, the pair stats for the latest two patches, so reasons can say e.g. *"Patch 16.18 to 16.20: paired with Ezreal 65× (5.0× chance), won 43%"*. That shows both the pro priority and the actual result. Re-run it after new data:

```bash
.venv\Scripts\python scripts\build_recent.py
```

### Bans: threat × damage

For each ban candidate, the app simulates the enemy picking it and measures how much **your** win chance drops: their counters to your picks, whole-team and lane matchups, and their players' comfort.
- **Ban score:** *enemy pick likelihood × that drop*.
- **Reasons** name the damage ("If they get it: −2.4% win chance for you") and what it beats ("Strong lane counter to your Syndra").

This matters because pros often don't protect their picks this way. Since Locke's release, Syndra's team banned an open Locke afterwards only 5% of the time, even though Locke won 62% of their lane matchups (48 games; 95% interval roughly 48–76%, so it could still be chance). A model that only imitates pros would never suggest it.

### Opponent mode

In setup, the opponent can draft **like the real team** (imitating its pick and ban habits, the default) or **smart** (taking one of the top recommendations, so it counters and protects its picks).

### Bot-lane duos

Pros pair ADC and support about 3× more selectively than any other role pair. The pick model has dedicated duo features: pairing with your already-picked duo partner, and that duo's win-rate lift. The ban model has one too: ban the partner the enemy's bot or support wants.

## Early game

`scripts/build_early.py` builds the early-game module. It forecasts the state of the game at 15 minutes from the draft and the players, and writes `models/early.pkl` and `reports/early_game.md`.

- **What it predicts (blue side's view):**
  - gold lead at 15 in each lane, each player against the opposite laner
  - team gold lead at 15
  - first dragon, void grubs majority, first blood and first tower
- **What it uses:**
  - each champion's lane-phase strength in its role
  - the specific lane matchup, beyond both champions' strength
  - each player's own early-game record
  - team early rating and objective habits
  - the jungler champion's objective rates
  - Elo
- **How it's tested:** the same no-look-ahead monthly backtest as the draft models.
- **In the app:** an *Early game forecast* panel above the board updates with every pick, showing lane bars and objective chances.
- **Team profiles:** `reports/team_early_profiles.csv` lists each team's average gold at 15, how often it converts a 1,500+ lead, and how often it comes back from a 1,500+ deficit.

```bash
.venv\Scripts\python scripts\build_early.py
```

## 15-minute game plan

`scripts/build_brief.py` fits how the gold lead at 15 turns into wins depending on **both** comps' timing. It writes `models/brief.pkl` and `reports/comp_timing.md`.

```bash
.venv\Scripts\python scripts\build_brief.py
```

- **Comp labels** come from the team's tag sheet `data/comp_labels.xlsx`, a copy of *LoL Comp Labels – Early Game Module*.
  - Each champion has a primary style, a secondary style and a timing.
  - A comp's timing score averages its five champions (Early = 0, Mid = 1, Late = 2). Its main and backup styles come from style points (primary = 2, secondary = 1).
  - Locke and Zaahen were added on 2026-11-12. Locke is tagged Early because the data shows a strong early mid (+384 gold at 15).
- **The model:** win ~ gold@15 + own timing + enemy timing + gold × own timing + gold × enemy timing + Elo. It is fitted on 2023–26 pro games and tested on the last 120 days.
- **What the 2023–26 data shows:**
  - Later comps can afford to be behind.
  - With timing only, an early comp facing a late comp needed about +1,100 gold at 15 to be a coin flip. Once comp **style** is included (the model the app uses), that drops to about **+345** at equal style: much of the apparent timing effect was style.
  - At the same gold at 15, teams that took first dragon or first tower went on to win more often. That is an association (objectives also reflect how the game is going), not a value you bank by taking them.
- **In the app:** the *15-minute game plan* panel appears once both teams have three picks. It shows:
  - both comps' labels
  - the break-even gold and the gold for a comfortable 65%
  - the early-game forecast
  - **bad (−2,500), even, good (+2,000) and best (+4,000)** scenarios, each with a win chance and what to do
  - a plan to minute 15, built from the sheet's Comp Types rows, the forecast's strongest and weakest lanes, and the likely objectives

**Comp style.** Each comp also gets a style mix (its share of each fight style) and a style-matchup score from the sheet's *Strong vs* / *Weak vs* columns.
- **Model choice:** `build_brief.py` scores four variants on the last 120 days and uses the best. Learned style-mix weights won (log-loss 0.5116 against 0.5153 for timing only).
- **What the data shows:** at the same gold at 15, poke/siege-heavy comps win less, and split-push and disengage comps win more.
- **Checking the sheet:** the report compares the sheet's style matchups with the data. It agrees on 10 of 12; it clearly disagrees that Poke beats Front-to-back.

**Win condition** (`draftlab/wincon.py`), shown in the brief:
- **Who needs to be doing well:**
  - It uses the team's win rate when that champion's lane is ahead against behind at 15, minus the usual gap for the role, combined with damage share.
  - Support and jungle gold mostly mirrors the whole game, so their raw gap isn't used on its own.
- **Game length** (the app shows this; the objective tendency is still computed in `wincon.py` and the report but hidden in the app for now):
  - These come from the 400 most similar comps (timing, style mix, enemy timing) over the last two years.
  - **Bias control:** early leads *cause* short games and objectives, so every number is compared with what an average comp with the **same gold at 15** did. Only differences beyond the noise are called out. The game-length tendency is "wants to end early", "can play long" or none; the objective tendency is drakes and soul, Baron, Atakhan or late Elder fights.
  - **Still an association:** what happens after minute 15 isn't in the data. For example, the raw "early comps win short games" disappears once early gold is accounted for, while early skirmish comps' lean on drakes remains.

## Tuning the weights

`scripts/tune.py` tunes how the model weights its data. It deletes games inside random patches and predicts who won them:

1. Patches are drawn at random (`--years 2026` by default). Each patch's games are split into 5 folds.
2. Each fold is deleted in turn. Every statistic is rebuilt without those games and without later patches. By default (`--chrono`) the known games are the patch's earliest ones and stats are as of the first deleted game, so nothing from the future leaks in. The older random mode (`--random`) keeps a random subset known, which lets later games of the same patch inform the prediction; its results (both tuning runs) found no improvement either way. Then the deleted games' winners are predicted from draft, players and Elo.
3. A logistic win model is scored **leave-one-patch-out**, so no patch's own results are used to fit the coefficients that score it.
4. A coordinate search adjusts the parameters on the *tune* patches. A grid is extended automatically when the best value sits on its edge. The result is then confirmed on separate *validation* patches. The parameters are:
   - meta half-life
   - matchup half-life
   - patch boost
   - fearless game decay
   - shrinkage strengths (`k_champ`, `k_pair`, `k_lane`)
   - Elo K
5. With `--apply`, tuned values are written to `params.json` **only if they also improve the validation patches**. `build.py` and the app then use them.

```bash
.venv\Scripts\python scripts\tune.py --years 2026 --apply
```
```bash
.venv\Scripts\python scripts\build.py
```

`draftlab/fast_stats.py` makes this feasible. Every statistic is a weighted sum over games, so each game's contributions are precomputed once, and any weighting or set of deleted games becomes a numpy `bincount`. Snapshots take about 1s instead of about 24s, with identical results.

### Missing results: shadow-variable bounds

`draftlab/bounds.py` adapts the partial-identification approach of Simchi-Levi et al. (bounds by linear programming with an AI prediction as a *shadow variable*) to a **binary outcome**: Team A wins or Team B wins.
- **Variables:** Y is the result, R says whether the result was observed, and Z is our model's binned win prediction.
- **Assumption:** Z ⟂ R | Y, meaning whether a result went missing doesn't depend on the prediction once you know who won.
- **Estimation:** two LPs give sharp lower and upper bounds on the true win rate. Slack minimisation absorbs finite-sample noise, and subsampling gives a confidence interval.

In simulation with wins deleted far more often than losses, the naive average was off by 18 points. The shadow interval contained the truth in every run and was about 45% narrower than the no-assumption bounds. Keep covariates X coarse: splitting into many Elo bins made Z only weakly informative within each bin and biased the LP (−0.11). `reports/tuning.md` repeats this test on real validation games.

## Independent review

`reports/judge_report.md` is an independent critique (agent definition in `.claude/agents/draft-lab-judge.md`). Main points and status:
- **Draft win effect is not significant** over Elo (log-loss gain 0.0006 ± 0.0022). The app now leads with pro-pick prediction and shows the draft effect only as "slight edge / no clear edge".
- **Counters are mostly noise.** Whole-team matchups are dropped from bans, and lane counters need 30+ games and show their sample size and "could be chance".
- **Elo across leagues:** league offsets added; training on the adjusted rating is still to do.
- **Still open:** a frozen final holdout with bootstrap intervals, champion strength adjusted for team strength, per-draft-stage validation, and LPL early-game coverage (about 15% of games).

## Limitations

- **Draft is a small part of winning.** The backtest compares "Elo only" with "Elo + draft" so the draft's real contribution is visible, not oversold.
- **Elo across regions** only becomes meaningful after international events.
- **Lane matchups and synergies are sparse in pro play.** They are heavily shrunk, and solo queue data would sharpen them (planned).
- **Roles for partial drafts are inferred** from each champion's role history, so flex picks can be mis-assigned until the draft fills out.

## Roadmap

- Solo queue champion pools per player (Riot API), to cover pocket picks.
- Early-game module: 10/15-minute gold, XP and CS benchmarks, and jungle pathing tendencies.
- Coach Comms assistant (LCK/LCP pilot): flag the moments where a 45-second call is worth spending.

## Credits

Match data is from [Oracle's Elixir](https://oracleselixir.com) by Tim Sevenhuysen ([support on Patreon](https://www.patreon.com/oracleselixir)). Champion icons come from Riot's Data Dragon. Game data is the property of Riot Games. This project isn't endorsed by Riot Games.

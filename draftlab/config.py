"""Project-wide settings."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Overridable so tests can run against synthetic data without touching the real folders.
RAW_DIR = Path(os.environ.get("DRAFTLAB_RAW_DIR", ROOT / "data" / "raw"))
MODEL_DIR = Path(os.environ.get("DRAFTLAB_MODEL_DIR", ROOT / "models"))
REPORT_DIR = Path(os.environ.get("DRAFTLAB_REPORT_DIR", ROOT / "reports"))

ROLES = ["top", "jng", "mid", "bot", "sup"]
ROLE_LABELS = {"top": "Top", "jng": "Jungle", "mid": "Mid", "bot": "Bot", "sup": "Support"}
SIDES = ("Blue", "Red")

# Leagues used to train/evaluate the draft models. Stats (champion, player, synergy)
# are built from *all* leagues in the data; these are the drafts we learn to imitate.
# Codes that don't appear in the data are simply ignored.
TRAIN_LEAGUES = {
    "LCK", "LPL", "LEC", "LCS", "LTA", "LTA N", "LTA S", "LCP", "CBLOL",
    "MSI", "WLDs", "FST", "EWC",
    "LFL", "PRM", "NACL", "LCKC", "LVP SL", "EM", "TCL", "NLC", "LJL", "VCS", "PCS",
}

# Recency weighting: a game this many days old counts half as much as one played today.
# Meta (presence, champion strength, player comfort, team habits): short memory, old seasons ~ignored.
HALF_LIFE_DAYS = 90
# Patch relevance for the meta memory: a game on the target patch gets weight x(1 + PATCH_BOOST),
# one patch away x(1 + PATCH_BOOST/2), two away x(1 + PATCH_BOOST/4), ... on top of recency.
# 10 chosen by the draft backtest (2026-11-12): vs 3, ban top-5 +2.4 pts (phase 1) and +1.2 (phase 2),
# picks +0.2, win prediction unchanged. Bans follow the current patch's power picks closely.
PATCH_BOOST = 10.0
# Fearless series: later games are drafted from leftover pools, so their picks/results say less
# about true priority and strength (60% of game-1 picks are top-30 champions vs 28% in game 5).
# Weight per game number in a fearless series; non-fearless games always weigh 1.
FEARLESS_GAME_WEIGHTS = {1: 1.0, 2: 0.8, 3: 0.6, 4: 0.45, 5: 0.35}
# Matchups (lane counters, synergies): long memory, so counters learned in past seasons carry over.
# With 540 days, a game from 3 years ago still counts ~1/4.
MATCHUP_HALF_LIFE_DAYS = 540

# Frozen final holdout (judge finding M4): the most recent ~6 months, every game from this date. No design
# choice may look at these games. build.py / build_brief.py make all choices on rolling-origin validation
# before this date and score the holdout only when run with --final-evaluation (user's explicit go-ahead).
HOLDOUT_START = "2026-05-12"

# App win model (A6). "a3a4" = A3 strength (strength_g15_tier_cal) + A4 draft layer from models/a3a4_win.json
# (scripts/build_a3a4.py); "legacy" = the original Elo + draft logistic model in models/draftlab.pkl. If the a3a4
# artifact is missing the app falls back to legacy. Brief and early forecasts keep using Elo (they are trained on it).
WIN_MODEL = os.environ.get("DRAFTLAB_WIN_MODEL", "a3a4")

# Team strength (M1). Values below were chosen on rolling-origin validation before HOLDOUT_START
# (see reports/backtest.md, "Variant selection").
ELO_REGRESS = 0.0            # share of the way back to the league mean after a break (0 = off)
ELO_REGRESS_GAP_DAYS = 30    # a break this long counts as a new split
ELO_LEAGUE_ADJUST = False    # train on Elo + leakage-free league offsets
# Champion strength (M8): True = per-champion effect from a logistic model with both teams' ratings as
# offsets; False = raw recency-weighted win rate.
ADJUSTED_STRENGTH = False

# Shrinkage strengths (pseudo-games pulling a win rate towards its prior).
K_CHAMP = 30.0
K_PAIR = 20.0
K_LANE = 15.0


def league_elo():
    """League strength offsets (scripts/build_league_elo.py): {"offsets": {league: elo}, "home": {team: league}}."""
    import json
    f = MODEL_DIR / "league_elo.json"
    if not f.exists():
        return {"offsets": {}, "home": {}}
    return json.loads(f.read_text(encoding="utf-8"))

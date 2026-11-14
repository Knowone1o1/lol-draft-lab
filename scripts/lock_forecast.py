"""Lock a pre-match forecast so it can be scored honestly afterwards (scripts/score_final.py).

Writes reports/forecast_lock_<date>/:
  forecast.json        series odds, score distribution, per-setup chances, draft predictions, baselines
  forecast.json.sha256 fingerprint of forecast.json -> post it publicly BEFORE the match as proof it wasn't edited
  models/              frozen copies of the exact models used (draftlab.pkl, a3a4_win.json, league_elo.json,
                       recent.pkl), so the scorecard uses them even after the app is rebuilt with the final's data

    .venv/Scripts/python scripts/lock_forecast.py --date 2026-11-14
"""
import argparse
import hashlib
import json
import pickle
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from draftlab.config import MODEL_DIR, REPORT_DIR  # noqa: E402

FROZEN = ["draftlab.pkl", "a3a4_win.json", "league_elo.json", "recent.pkl"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def series_summary(G, a, b):
    S = G.groupby("series").agg(winner=("winner", "last"), score=("score_after", "last"), games=("game", "max"))
    n = len(S)
    return {"p_a_series": round(float((S["winner"] == a).mean()), 4), "n_series": n,
            "score_dist": {k: round(v / n, 4) for k, v in S["score"].value_counts().sort_index().items()},
            "p_a_per_game": round(float((G["winner"] == a).mean()), 4),
            "p_goes_to_5": round(float((S["games"] == 5).mean()), 4)}


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", default="2026-11-14")
    ap.add_argument("--sim", default=str(REPORT_DIR / "sim_GenG_vs_Bilibili_Gaming_a3a4.pkl"))
    ap.add_argument("--baseline-sim", default=str(REPORT_DIR / "sim_GenG_vs_Bilibili_Gaming.pkl"))
    a = ap.parse_args()
    out = REPORT_DIR / f"forecast_lock_{a.date}"
    if (out / "forecast.json").exists():
        sys.exit(f"{out / 'forecast.json'} already exists: a locked forecast is never overwritten.")
    (out / "models").mkdir(parents=True, exist_ok=True)
    for f in FROZEN:
        shutil.copy2(MODEL_DIR / f, out / "models" / f)

    d = pickle.load(open(a.sim, "rb"))
    G, A, B = d["games"], d["a"], d["b"]
    base = pickle.load(open(a.baseline_sim, "rb"))
    from draftlab.winmodels.runtime import A3A4WinModel
    rt = A3A4WinModel(out / "models" / "a3a4_win.json")
    setups = {f"{A} blue, {'blue' if bf else 'red'} picks first": round(float(1 / (1 + 2.718281828 ** -rt.strength_logit(A, B, bf))), 4)
              for bf in (True, False)}
    setups.update({f"{A} red, {'blue' if bf else 'red'} picks first": round(float(1 - 1 / (1 + 2.718281828 ** -rt.strength_logit(B, A, bf))), 4)
                   for bf in (True, False)})

    g1 = G[G["game"] == 1]
    def top(counter, k=5):
        return [{"champion": c, "share": round(v / len(g1), 3)} for c, v in counter.most_common(k)]
    draft = {}
    for team in (A, B):
        draft[team] = {
            "game1_first_ban": top(Counter(r.bans[team][0] for r in g1.itertuples() if r.bans[team])),
            "game1_first_pick": top(Counter(r.picks[team][0] for r in g1.itertuples() if r.picks[team])),
            "series_pick_rate": [{"champion": c, "share_of_series": round(v / G["series"].nunique(), 3)}
                                 for c, v in Counter(c for _, x in G.groupby("series")
                                                     for c in set(sum((p[team] for p in x["picks"]), []))).most_common(8)],
        }
    fc = {
        "match": f"{A} vs {B}", "event": "Worlds 2026 Grand Final (Bo5, fearless)", "match_date": a.date,
        "locked_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "team_a": A, "team_b": B,
        "model": {"name": "A3 strength + A4 draft layer (app WIN_MODEL=a3a4)", "built": rt.art["built"],
                  "data_to": rt.art["data_to"], "simulations": len(G.groupby("series")),
                  "form_sd": d["args"].get("form_sd"), "seed": d["args"].get("seed")},
        "forecast": series_summary(G, A, B),
        "pre_draft_chance_by_setup": setups,
        "draft_predictions": draft,
        "baselines": {"legacy_app_model": series_summary(base["games"], A, B), "coin_flip": {"p_a_series": 0.5, "p_a_per_game": 0.5}},
        "frozen_model_sha256": {f: sha(out / "models" / f) for f in FROZEN},
        "simulation_sha256": {"model": sha(a.sim), "legacy_baseline": sha(a.baseline_sim)},
    }
    text = json.dumps(fc, indent=1, ensure_ascii=False)
    (out / "forecast.json").write_text(text, encoding="utf-8")
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    (out / "forecast.json.sha256").write_text(f"{h}  forecast.json\n", encoding="utf-8")
    f = fc["forecast"]
    print(f"Locked {fc['match']} at {fc['locked_at_utc']}: {A} {f['p_a_series']:.1%} series, {f['p_a_per_game']:.1%} per game")
    print(f"SHA-256 of forecast.json: {h}")
    print(f"Folder: {out}")


if __name__ == "__main__":
    main()

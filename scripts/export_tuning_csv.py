"""Turn reports/tuning_data.json into flat CSVs for charts (reports/tuning_csv/)."""
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.config import REPORT_DIR  # noqa: E402

DESCRIPTIONS = {
    "half_life": "Meta memory: days until a game counts half (pick/ban priority, strength, comfort)",
    "matchup_half_life": "Counter/synergy memory: days until a game counts half",
    "patch_boost": "Extra weight for games on or near the target patch",
    "fearless_decay": "Weight of each later fearless game vs the one before (game n = decay^(n-1))",
    "k_champ": "Shrinkage of champion win rates (pseudo-games at 50%)",
    "k_pair": "Shrinkage of synergies and whole-team matchups",
    "k_lane": "Shrinkage of lane counters",
    "elo_k": "Elo update size per game",
}


def write(path, rows, cols):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: ("" if r.get(c) is None else r.get(c)) for c in cols})


def main():
    d = json.loads((REPORT_DIR / "tuning_data.json").read_text(encoding="utf-8"))
    out = REPORT_DIR / "tuning_csv"
    out.mkdir(exist_ok=True)
    write(out / "trials.csv", d["trials"],
          ["trial", "pass", "param", "value", "log_loss", "acc", "accepted", "best_log_loss", "best_acc",
           "val_log_loss", "val_acc"])
    rows = []
    for label, splits in d["known_curve"].items():
        for split, pts in splits.items():
            for p in pts:
                rows.append({"weights": label, "split": split, "known_mid": round((p["known_from"] + p["known_to"]) / 2, 3), **p})
    write(out / "known_curve.csv", rows, ["weights", "split", "known_from", "known_to", "known_mid", "n", "acc", "log_loss"])
    write(out / "params.csv", [{"parameter": k, "start": d["start"].get(k), "tuned": d["tuned"].get(k),
                                "meaning": DESCRIPTIONS.get(k, "")} for k in DESCRIPTIONS],
          ["parameter", "start", "tuned", "meaning"])
    rows = [{"weights": lab, "split": sp, **m} for lab, s in d["summary"].items() for sp, m in s.items()]
    write(out / "summary.csv", rows, ["weights", "split", "log_loss", "auc", "acc", "n"])
    write(out / "bounds.csv", [{"pattern": b["pattern"], "truth": b["truth"], "naive": b["naive"],
                                "ci_lower": b["ci"][0], "ci_upper": b["ci"][1],
                                "worst_lower": b["worst"][0], "worst_upper": b["worst"][1]} for b in d["bounds"]],
          ["pattern", "truth", "naive", "ci_lower", "ci_upper", "worst_lower", "worst_upper"])
    print(f"wrote CSVs to {out}: patches tune={d['tune_patches']} val={d['val_patches']}")


if __name__ == "__main__":
    main()

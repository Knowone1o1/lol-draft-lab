"""Evaluate pre-game win models walk-forward and write a comparable result set to reports/runs/.

    .venv\\Scripts\\python scripts\\evaluate.py                                  # baselines, 2024-01-01 .. holdout
    .venv\\Scripts\\python scripts\\evaluate.py --models always_blue elo elo_league --start 2025-01-01
    .venv\\Scripts\\python scripts\\evaluate.py --final-evaluation              # ONLY when the user says so

The frozen holdout (config.HOLDOUT_START onward) is never scored without --final-evaluation.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab.eval import evaluate, load_eval_games  # noqa: E402
from draftlab.winmodels import REGISTRY, get_model  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=["always_blue", "elo", "elo_league"], choices=sorted(REGISTRY))
    ap.add_argument("--start", default="2024-01-01", help="first day evaluated (training always uses all earlier games)")
    ap.add_argument("--baseline", default="elo")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--draft-base", default=None, help="strength model used as the fixed offset by draft_* models (A4)")
    ap.add_argument("--untuned", action="store_true", help="ignore reports/runs/<model>_tuned.json")
    ap.add_argument("--final-evaluation", action="store_true", help="score the frozen holdout (only on the user's word)")
    a = ap.parse_args()
    t0 = time.time()
    games = load_eval_games()
    models = []
    for n in a.models:
        tuned = Path("reports/runs") / f"{n}_tuned.json"     # e.g. strength_tuned.json from scripts/tune_strength.py
        if n.startswith("strength") and not tuned.exists() and not a.untuned:
            sys.exit(f"{tuned} missing: run scripts/tune_strength.py --model {n} first, or pass --untuned")
        params = json.loads(tuned.read_text())["params"] if tuned.exists() and not a.untuned else {}
        if n.startswith("draft_") and a.draft_base:
            params = {**params, "base": a.draft_base}
        print(f"{n}: {'tuned ' + str(params) if params else 'default params'}", flush=True)
        models.append(get_model(n, **params))
    res = evaluate(games, models, start=a.start, final=a.final_evaluation, baseline=a.baseline, B=a.boot,
                   seed=a.seed, note=a.note)
    print(f"Run {res.run_id} in {time.time() - t0:.0f}s -> reports/runs/{res.run_id}.md")
    print((Path("reports/runs") / f"{res.run_id}.md").read_text(encoding="utf-8")[:3000])


if __name__ == "__main__":
    main()

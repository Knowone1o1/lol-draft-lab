"""Download pro players' listed solo-queue accounts from Leaguepedia's public Cargo API, slowly.

Anonymous access only. One request at a time, a long pause between requests, exponential
back-off when the wiki answers 'ratelimited', every response cached under
data/soloq/raw/leaguepedia/ so a re-run continues where it stopped. After too many refusals
in a row it stops (no workarounds).

    .venv\\Scripts\\python scripts\\soloq_leaguepedia.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab.soloq import SOLOQ_DIR  # noqa: E402

API = "https://lol.fandom.com/api.php"
CACHE = SOLOQ_DIR / "raw" / "leaguepedia"
PAGE = 500
FIELDS = ("Players.ID=ID,Players.OverviewPage=Page,Players.Name=Name,Players.Team=Team,"
          "Players.Role=Role,Players.Residency=Residency,Players.SoloqueueIds=SoloqueueIds,"
          "Players.IsRetired=IsRetired")
WHERE = "Players.SoloqueueIds IS NOT NULL AND Players.SoloqueueIds != ''"
UA = "LoLDraftLab-research/0.1 (personal non-commercial project; slow crawl)"


def fetch(offset: int) -> dict:
    params = {"action": "cargoquery", "format": "json", "tables": "Players", "fields": FIELDS,
              "where": WHERE, "order_by": "Players.OverviewPage", "limit": PAGE, "offset": offset}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pause", type=float, default=30, help="seconds between successful requests")
    ap.add_argument("--first-backoff", type=float, default=120, help="first wait after a refusal (doubles)")
    ap.add_argument("--max-refusals", type=int, default=6, help="give up after this many refusals in a row")
    ap.add_argument("--max-pages", type=int, default=60)
    args = ap.parse_args()
    CACHE.mkdir(parents=True, exist_ok=True)

    refusals = 0
    for page in range(args.max_pages):
        f = CACHE / f"players_{page * PAGE:06d}.json"
        if f.exists():
            rows = json.loads(f.read_text(encoding="utf-8")).get("cargoquery", [])
            if len(rows) < PAGE:
                print(f"page {page}: cached, last page ({len(rows)} rows). Done.", flush=True)
                return 0
            continue
        while True:
            try:
                data = fetch(page * PAGE)
            except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as e:
                data = {"error": {"code": f"http:{getattr(e, 'code', e)}"}}
            err = (data.get("error") or {}).get("code")
            if not err:
                break
            refusals += 1
            if refusals > args.max_refusals:
                print(f"STOPPED: {refusals} refusals in a row (last: {err}) at page {page}. "
                      "Re-run later to continue from here.", flush=True)
                return 2
            wait = args.first_backoff * 2 ** (refusals - 1)
            print(f"page {page}: refused ({err}), waiting {wait:.0f}s [{refusals}/{args.max_refusals}]", flush=True)
            time.sleep(wait)
        refusals = 0
        f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        n = len(data.get("cargoquery", []))
        print(f"page {page}: {n} rows saved", flush=True)
        if n < PAGE:
            print("Done.", flush=True)
            return 0
        time.sleep(args.pause)
    print("Reached --max-pages.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

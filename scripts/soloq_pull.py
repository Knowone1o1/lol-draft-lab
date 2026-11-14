"""Pull pro players' ranked solo-queue matches from the official Riot API.

Reads data/soloq/pro_accounts.csv (from soloq_accounts.py) and RIOT_API_KEY from .env.
Resumable: re-run the same command after an interruption or a key refresh.

    .venv\\Scripts\\python scripts\\soloq_pull.py                       # first pull: 20 most recent games per account
    .venv\\Scripts\\python scripts\\soloq_pull.py --per-account 100 --days 365
    .venv\\Scripts\\python scripts\\soloq_pull.py --leagues LCK LEC     # only some leagues

Budget with a development key (100 requests / 2 min ~ 3,000 / hour):
  1 request per account to resolve the PUUID, 1 per 100 match ids, 1 per new match.
  Matches shared by two tracked pros are fetched once.
Match-v5 only keeps about two years of history.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab import soloq as sq  # noqa: E402

PUUID_FIELDS = ["riot_id", "server", "player", "puuid", "status", "resolved_at"]


def resolve_puuids(client, accounts, verbose=True):
    done = {(r["riot_id"], r["server"]): r for r in sq.read_csv(sq.PUUIDS_CSV)}
    todo = [a for a in accounts if (a["riot_id"], a["server"]) not in done]
    print(f"PUUIDs: {len(done)} cached, {len(todo)} to resolve")
    for i, a in enumerate(todo, 1):
        parts = sq.split_riot_id(a["riot_id"])
        status, puuid = "bad_riot_id", ""
        if parts:
            # account-v1 is global: any regional route works; use the account's own when known.
            res = client.account_by_riot_id(*parts, region=sq.region_for(a["server"]) or "asia")
            if res and res.get("puuid"):
                status, puuid = "ok", res["puuid"]
            else:
                status = "not_found"   # renamed account, typo, or a stale ID on the source page
        row = {"riot_id": a["riot_id"], "server": a["server"], "player": a["player"],
               "puuid": puuid, "status": status,
               "resolved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        sq.append_csv(sq.PUUIDS_CSV, [row], PUUID_FIELDS)
        done[(a["riot_id"], a["server"])] = row
        if verbose and (i % 25 == 0 or i == len(todo)):
            print(f"  resolved {i}/{len(todo)}")
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-account", type=int, default=20,
                    help="most recent ranked games to keep per account (default 20)")
    ap.add_argument("--days", type=int, default=730, help="only games from the last N days (default 730)")
    ap.add_argument("--leagues", nargs="*", help="only pros from these leagues (pro_accounts.csv 'league')")
    ap.add_argument("--refresh-ids", action="store_true",
                    help="re-download match id lists (to pick up games played since the last run)")
    ap.add_argument("--max-requests", type=int, default=0, help="stop after this many API calls (0 = no cap)")
    args = ap.parse_args()

    accounts = [a for a in sq.read_csv(sq.ACCOUNTS_CSV) if a.get("riot_id")]
    if not accounts:
        sys.exit(f"{sq.ACCOUNTS_CSV} is missing or empty: run scripts/soloq_accounts.py first.")
    if args.leagues:
        keep = {l.upper() for l in args.leagues}
        accounts = [a for a in accounts if (a.get("league") or "").upper() in keep]
    accounts = [a for a in accounts if sq.region_for(a["server"])]
    print(f"{len(accounts)} accounts with a supported server")

    client = sq.RiotClient(sq.load_api_key())
    resolved = resolve_puuids(client, accounts)

    # puuid -> (player, server); one pro can have several accounts.
    wanted = {}
    for a in accounts:
        r = resolved.get((a["riot_id"], a["server"]))
        if r and r["status"] == "ok":
            wanted[r["puuid"]] = (a["player"], a["server"])
    print(f"{len(wanted)} accounts resolved to a PUUID")

    # ---- match ids
    sq.MATCH_IDS_DIR.mkdir(parents=True, exist_ok=True)
    start_time = int(time.time()) - args.days * 86400
    ids_by_puuid = {}
    for i, (puuid, (player, server)) in enumerate(wanted.items(), 1):
        f = sq.MATCH_IDS_DIR / f"{puuid}.txt"
        if f.exists() and not args.refresh_ids:
            ids = f.read_text(encoding="utf-8").split()
        else:
            ids = client.ranked_match_ids(puuid, sq.region_for(server), start_time, args.per_account)
            f.write_text("\n".join(ids), encoding="utf-8")
        ids_by_puuid[puuid] = ids[: args.per_account]
        if i % 25 == 0 or i == len(wanted):
            print(f"  match ids: {i}/{len(wanted)} accounts")
        if args.max_requests and client.n_requests >= args.max_requests:
            print("Request cap reached; re-run to continue.")
            return

    # ---- match details (each match fetched once, rows for every tracked pro in it)
    have = {(r["match_id"], r["puuid"]) for r in sq.read_csv(sq.MATCHES_CSV)}
    have_matches = {m for m, _ in have} | {r["match_id"] for r in sq.read_csv(sq.FAILED_CSV)}
    todo: dict[str, str] = {}   # match id -> region
    for puuid, ids in ids_by_puuid.items():
        for mid in ids:
            if mid not in have_matches:
                todo.setdefault(mid, sq.region_for(wanted[puuid][1]))
    print(f"Matches: {len(have_matches)} already stored, {len(todo)} to fetch "
          f"(~{len(todo) / 50:.0f} min at dev-key speed)")
    t0 = time.time()
    for i, (mid, region) in enumerate(todo.items(), 1):
        m = client.match(mid, region)
        if not m:
            sq.append_csv(sq.FAILED_CSV, [{"match_id": mid, "reason": "not_found"}], ["match_id", "reason"])
            continue
        sq.append_csv(sq.MATCHES_CSV, sq.summarize_match(m, wanted), sq.MATCH_FIELDS)
        if i % 50 == 0 or i == len(todo):
            el = time.time() - t0
            print(f"  {i}/{len(todo)} matches, {el / 60:.1f} min, ~{(len(todo) - i) * el / i / 60:.0f} min left")
        if args.max_requests and client.n_requests >= args.max_requests:
            print("Request cap reached; re-run to continue.")
            return
    print(f"Done. {client.n_requests} API requests this run.")


if __name__ == "__main__":
    main()

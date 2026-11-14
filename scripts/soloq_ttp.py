"""Pro players' solo-queue games from TrackingThePros (trackingthepros.com), slowly and cached.

TrackingThePros lists ~2,300 pros with their known accounts and a feed of their recent ranked
games. robots.txt allows everything; the site's Terms of Service link is a 404 (checked 2026-11-12).
No Riot API key is needed. Politeness: one request at a time, --pause seconds apart (default 5),
back-off on errors, stop after repeated failures, every response cached under data/soloq/raw/ttp/
so nothing is fetched twice and runs resume where they stopped.

    .venv\\Scripts\\python scripts\\soloq_ttp.py players             # 1 request: the pro list -> match to OE names
    .venv\\Scripts\\python scripts\\soloq_ttp.py feed --pages 1      # 20 most recent games per matched pro
    .venv\\Scripts\\python scripts\\soloq_ttp.py feed --pages 10 --days 120 --leagues LCK LPL LEC
    .venv\\Scripts\\python scripts\\soloq_ttp.py build               # cached pages -> CSVs + coverage report

Outputs (data/soloq/): ttp_players.csv (OE player <-> TTP id), pro_accounts.csv (accounts seen),
matches.csv (one row per solo-queue game), and reports/soloq_coverage.md.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab.config import REPORT_DIR  # noqa: E402
from draftlab.soloq import SOLOQ_DIR, norm_name, oe_players  # noqa: E402

BASE = "https://www.trackingthepros.com"
RAW = SOLOQ_DIR / "raw" / "ttp"
FEED = RAW / "feed"
LIST_JSON = RAW / "list_players.json"
TTP_PLAYERS_CSV = SOLOQ_DIR / "ttp_players.csv"
UA = "LoLDraftLab-research/0.1 (personal non-commercial project; slow crawl)"

# Which TTP regions are plausible for a player whose latest pro game was in this league.
LEAGUE_REGIONS = {
    "LCK": {"KR"}, "LCKC": {"KR"}, "LPL": {"CN", "KR"},
    "LEC": {"EU", "KR", "TR", "RU"}, "LFL": {"EU"}, "PRM": {"EU"}, "LVP SL": {"EU"}, "NLC": {"EU"},
    "EM": {"EU", "TR", "RU"}, "TCL": {"TR", "EU"},
    "LCS": {"NA", "KR", "EU"}, "LTA N": {"NA", "KR", "EU"}, "NACL": {"NA", "EU"},
    "CBLOL": {"BR", "LLA", "KR"}, "LTA S": {"BR", "LLA", "KR"},
    "LCP": {"LCP", "VT", "JP", "OCE", "KR"}, "PCS": {"LCP", "OCE", "JP", "KR"},
    "VCS": {"VT", "LCP"}, "LJL": {"JP", "KR", "LCP"},
}
# Fetch order: the leagues the draft models care most about first.
LEAGUE_PRIORITY = ["LCK", "LPL", "LEC", "LCS", "LTA N", "LTA S", "LTA", "LCP", "CBLOL", "WLDs", "MSI",
                   "EWC", "FST", "LCKC", "PRM", "LFL", "NACL", "LVP SL", "TCL", "NLC", "VCS", "PCS", "LJL", "EM"]


class Polite:
    def __init__(self, pause: float, max_failures: int = 5):
        self.pause, self.max_failures, self.failures, self.last, self.n = pause, max_failures, 0, 0.0, 0

    def get_json(self, url: str):
        while True:
            wait = self.pause - (time.monotonic() - self.last)
            if wait > 0:
                time.sleep(wait)
            self.last = time.monotonic()
            self.n += 1
            req = urllib.request.Request(url, headers={"User-Agent": UA, "X-Requested-With": "XMLHttpRequest"})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    body = r.read().decode("utf-8")
                data = json.loads(body)
                self.failures = 0
                return data
            except urllib.error.HTTPError as e:
                code = e.code
                if code == 500:         # the feed answers 500 for some 'before' values: treat as end of feed
                    return None
                err = f"HTTP {code}"
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                err = type(e).__name__
            self.failures += 1
            if self.failures >= self.max_failures:
                raise SystemExit(f"STOPPED after {self.failures} failures in a row (last: {err}). "
                                 "Re-run later to resume.")
            back = 60 * 2 ** (self.failures - 1)
            print(f"  {err}; waiting {back}s [{self.failures}/{self.max_failures}]", flush=True)
            time.sleep(back)


# ---------------------------------------------------------------- players

def cmd_players(args):
    RAW.mkdir(parents=True, exist_ok=True)
    if not LIST_JSON.exists() or args.refresh:
        data = Polite(args.pause).get_json(f"{BASE}/d/list_players?filter_region=ALL&filter_role=ALL&filter_subs=1")
        LIST_JSON.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    ttp = pd.DataFrame(json.loads(LIST_JSON.read_text(encoding="utf-8"))["data"])
    ttp["k"] = ttp["name"].map(norm_name)
    oe = oe_players()
    oe["k"] = oe["player"].map(norm_name)

    rows = []
    for _, p in oe.iterrows():
        cands = ttp[ttp["k"] == p["k"]]
        if cands.empty:
            continue
        ok = LEAGUE_REGIONS.get(p["league"])
        if ok is not None:
            in_region = cands[cands["current_region"].isin(ok) | cands["home_region"].isin(ok)]
            region_ok = not in_region.empty
            cands = in_region if region_ok else cands
        else:
            region_ok = True     # international events: any region is plausible
        if len(cands) > 1:       # same handle, several people: keep the one with the most accounts
            cands = cands.sort_values("player_accounts", ascending=False)
        c = cands.iloc[0]
        rows.append({
            "player": p["player"], "league": p["league"], "team": p["team"], "role": p["role"],
            "last_pro_game": str(p["last_pro_game"])[:10], "oe_games": p["oe_games"],
            "ttp_id": int(c["id"]), "ttp_name": c["name"], "ttp_team": c.get("team_name"),
            "ttp_region": c["current_region"], "ttp_position": c["position"],
            "ttp_accounts": c.get("player_accounts"), "ttp_rank": c.get("rankHigh"),
            "ttp_top_champ_ids": "|".join(str(c.get(f"champ_{i}") or "") for i in (1, 2, 3)),
            "ttp_top_champ_pct": "|".join(str(c.get(f"champ_{i}_per") or "") for i in (1, 2, 3)),
            "region_ok": int(region_ok), "n_same_name": len(ttp[ttp["k"] == p["k"]]),
        })
    out = pd.DataFrame(rows)
    out.to_csv(TTP_PLAYERS_CSV, index=False, encoding="utf-8")
    oe["matched"] = oe["player"].isin(set(out["player"]))
    oe.drop(columns="k").to_csv(SOLOQ_DIR / "oe_players_match_status.csv", index=False, encoding="utf-8")
    print(f"TTP pros: {len(ttp)}; OE training-league players 2025-26: {len(oe)}; matched: {len(out)} "
          f"({int((out['region_ok'] == 0).sum())} with an unexpected region, kept but flagged)")


# ---------------------------------------------------------------- feed

def feed_pages(ttp_id: int):
    """Cached feed pages for one pro, newest first: list of (before, items)."""
    pages = []
    for f in FEED.glob(f"{ttp_id}_*.json"):
        before = int(f.stem.split("_")[1])
        d = json.loads(f.read_text(encoding="utf-8"))
        pages.append((before, (d or {}).get("data") or []))
    return sorted(pages, key=lambda x: (x[0] != 0, -x[0]))


def cmd_feed(args):
    FEED.mkdir(parents=True, exist_ok=True)
    pl = pd.read_csv(TTP_PLAYERS_CSV)
    if args.leagues:
        pl = pl[pl["league"].isin(args.leagues)]
    if not args.include_flagged:
        pl = pl[pl["region_ok"] == 1]
    pri = {l: i for i, l in enumerate(LEAGUE_PRIORITY)}
    pl = pl.assign(_p=pl["league"].map(pri).fillna(99)).sort_values(["_p", "oe_games"], ascending=[True, False])
    pl = pl.drop_duplicates("ttp_id")
    cutoff = time.time() - args.days * 86400
    http = Polite(args.pause)
    print(f"{len(pl)} pros; up to {args.pages} page(s) of 20 games each, back to {args.days} days", flush=True)
    t0 = time.time()
    for i, (_, p) in enumerate(pl.iterrows(), 1):
        tid = int(p["ttp_id"])
        cached = {b: items for b, items in feed_pages(tid)}
        before, n = 0, 0
        while n < args.pages:
            if before in cached:
                items = cached[before]
            else:
                d = http.get_json(f"{BASE}/d/pro_feed?player_id={tid}&type[]=soloq&"
                                  + (f"before={before}" if before else ""))
                (FEED / f"{tid}_{before}.json").write_text(json.dumps(d or {"data": []}, ensure_ascii=False),
                                                           encoding="utf-8")
                items = (d or {}).get("data") or []
            n += 1
            if len(items) < 20:
                break
            oldest = min(x["card_timestamp"] for x in items)
            if oldest < cutoff:
                break
            before = oldest
        if i % 25 == 0 or i == len(pl):
            el = time.time() - t0
            print(f"  {i}/{len(pl)} pros, {http.n} requests, {el / 60:.1f} min", flush=True)
    print(f"Done: {http.n} requests.", flush=True)


# ---------------------------------------------------------------- build

def champ_names_by_id():
    """Riot champion numeric id -> name, from Data Dragon (Riot's public static CDN; cached)."""
    f = RAW / "ddragon_champions.json"
    if not f.exists():
        try:
            ver = json.loads(urllib.request.urlopen("https://ddragon.leagueoflegends.com/api/versions.json",
                                                    timeout=30).read())[0]
            time.sleep(1)
            champs = json.loads(urllib.request.urlopen(
                f"https://ddragon.leagueoflegends.com/cdn/{ver}/data/en_US/champion.json", timeout=30).read())
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"  (champion names unavailable: {e}; top-3 champions left as ids)")
            return {}
        f.write_text(json.dumps({v["key"]: v["name"] for v in champs["data"].values()}, ensure_ascii=False),
                     encoding="utf-8")
    d = json.loads(f.read_text(encoding="utf-8"))
    if "data" in d:      # raw champion.json saved by hand
        d = {v["key"]: v["name"] for v in d["data"].values()}
    return d


def cmd_build(args):
    pl = pd.read_csv(TTP_PLAYERS_CSV)
    by_id = pl.drop_duplicates("ttp_id").set_index("ttp_id")
    rows = {}
    for f in FEED.glob("*.json"):
        tid = int(f.stem.split("_")[0])
        if tid not in by_id.index:
            continue
        for x in (json.loads(f.read_text(encoding="utf-8")) or {}).get("data") or []:
            if x.get("card_type") != "soloq":
                continue
            c = x["card_data"]
            k, d, a = (list(map(int, c["game_kda"].replace(" ", "").split("/"))) + [None] * 3)[:3] \
                if c.get("game_kda") else (None, None, None)
            rows[x["card_id"]] = {
                "card_id": x["card_id"], "player": by_id.loc[tid, "player"], "ttp_id": tid,
                "league": by_id.loc[tid, "league"], "team": by_id.loc[tid, "team"], "role": by_id.loc[tid, "role"],
                "date_utc": x["card_date"], "server": c.get("account_server"), "account": c.get("account_name"),
                "champion": c.get("game_champion"), "win": int(c.get("game_result") == "win"),
                "kills": k, "deaths": d, "assists": a, "length_min": c.get("game_length"),
            }
    m = pd.DataFrame(rows.values())
    if m.empty:
        sys.exit("No cached feed pages yet: run the 'feed' step first.")
    m = m.sort_values(["player", "date_utc"])
    m.to_csv(SOLOQ_DIR / "matches.csv", index=False, encoding="utf-8")

    acc = (m.groupby(["player", "team", "league", "role", "server", "account"])
             .agg(games=("card_id", "size"), first_seen=("date_utc", "min"), last_seen=("date_utc", "max"))
             .reset_index().rename(columns={"account": "riot_id"}))
    acc["source"] = "trackingthepros"
    acc.to_csv(SOLOQ_DIR / "pro_accounts.csv", index=False, encoding="utf-8")

    names = champ_names_by_id()
    pl["ttp_top_champs"] = pl["ttp_top_champ_ids"].fillna("").map(
        lambda s: "|".join(names.get(i, "") for i in str(s).split("|") if i and i != "nan"))
    pl.to_csv(TTP_PLAYERS_CSV, index=False, encoding="utf-8")
    write_report(pl, m, acc)
    print(f"{len(m)} games for {m['player'].nunique()} pros on {len(acc)} accounts -> data/soloq/matches.csv")


def write_report(pl, m, acc):
    oe = pd.read_csv(SOLOQ_DIR / "oe_players_match_status.csv")
    g = m.groupby("player").agg(games=("card_id", "size"), first=("date_utc", "min"), last=("date_utc", "max"))
    lg = oe.groupby("league").agg(oe_players=("player", "size"), mapped=("matched", "sum"))
    have = m.groupby("league").agg(with_games=("player", "nunique"), games=("card_id", "size"))
    lg = lg.join(have).fillna(0).astype(int)
    lg["mapped_pct"] = (100 * lg["mapped"] / lg["oe_players"]).round(0).astype(int)
    pri = {l: i for i, l in enumerate(LEAGUE_PRIORITY)}
    lg = lg.assign(_p=lg.index.map(lambda x: pri.get(x, 99))).sort_values("_p").drop(columns="_p")
    srv = m.groupby("server").size().sort_values(ascending=False)
    lpl = m[m["league"] == "LPL"].groupby("server").size()
    span_days = (pd.to_datetime(g["last"]) - pd.to_datetime(g["first"])).dt.days

    lines = [
        "# Pro solo-queue history: coverage",
        "",
        f"Built {time.strftime('%Y-%m-%d')} by `scripts/soloq_ttp.py build` from cached TrackingThePros pages.",
        "",
        "## Source and terms",
        "",
        "- **TrackingThePros** (trackingthepros.com): pro list with known accounts and a feed of recent ranked games."
        " robots.txt allows all crawling. The Terms of Service link returned 404 on 2026-11-12, so no terms could be read."
        " Crawled one request at a time, 5 s apart, every response cached under `data/soloq/raw/ttp/`."
        " It's a personal, non-commercial project; ask the site owner before publishing the raw data.",
        "- Tried first and dropped: protracker.gg (parked domain), dpm.lol (Cloudflare challenge on every page, not bypassed),"
        " Leaguepedia API (anonymous queries rate-limited), official Riot API (no key).",
        "",
        "## Coverage per league",
        "",
        "Players = everyone who played in that league in 2025-26 per Oracle's Elixir (by the league of their latest game).",
        "Mapped = found on TrackingThePros by name. With games = at least one solo-queue game collected.",
        "",
        "| League | OE players | Mapped | % | With games | Games |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for l, r in lg.iterrows():
        lines.append(f"| {l} | {r.oe_players} | {r.mapped} | {r.mapped_pct}% | {r.with_games} | {r.games} |")
    lines += [
        f"| **Total** | {lg.oe_players.sum()} | {lg.mapped.sum()} | {round(100 * lg.mapped.sum() / lg.oe_players.sum())}% "
        f"| {lg.with_games.sum()} | {lg.games.sum()} |",
        "",
        f"- {len(m)} games, {m['player'].nunique()} pros, {len(acc)} accounts, from {m['date_utc'].min()[:10]} to {m['date_utc'].max()[:10]}.",
        f"- Games per pro: median {int(g['games'].median())}, history span per pro: median {int(span_days.median())} days.",
        "- Games by server: " + ", ".join(f"{s} {n}" for s, n in srv.items()) + ".",
        "- LPL pros' games by server: " + (", ".join(f"{s} {n}" for s, n in lpl.items()) or "none") + ".",
        "",
        "## What's missing",
        "",
        "- **Chinese servers.** TrackingThePros, like every Riot-API-based site, can't see Chinese-server games."
        " LPL pros are covered only when they also play on Korean (or other) accounts, mostly imports and bootcamps.",
        "- **Depth.** The feed is paged 20 games at a time, newest first. This pull keeps only recent games, not two"
        " years: enough for current comfort and pocket picks, not for back-testing old drafts.",
        "- **Smaller leagues.** EM, LJL, VCS, NLC and PCS players are often not on TrackingThePros.",
        "- **Name matching.** Players are matched on their handle. A handle shared by two pros is resolved by region;"
        " matches with an unexpected region are flagged (`region_ok=0` in `ttp_players.csv`) and skipped.",
        "",
        "## Files",
        "",
        "- `data/soloq/ttp_players.csv`: Oracle's Elixir player ↔ TrackingThePros id, region, rank, top-3 solo-queue champions.",
        "- `data/soloq/pro_accounts.csv`: accounts seen in the games (server, Riot ID, games, first/last seen).",
        "- `data/soloq/matches.csv`: one row per game (date, account, champion, win, KDA, length).",
        "- `data/soloq/oe_players_match_status.csv`: every OE training-league player and whether they were mapped.",
        "",
        "## Next step (not done)",
        "",
        "Turn this into draft features: per player, the share of recent solo-queue games on each champion"
        " (as of the day before each pro game), plus a 'practised recently, never played on stage' flag for pocket picks."
        " Test whether it improves pick top-5 in the backtest before keeping it.",
    ]
    (REPORT_DIR / "soloq_coverage.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("players")
    a.add_argument("--refresh", action="store_true")
    a.add_argument("--pause", type=float, default=5)
    b = sub.add_parser("feed")
    b.add_argument("--pages", type=int, default=1, help="pages of 20 games per pro (default 1)")
    b.add_argument("--days", type=int, default=365, help="stop paging past games older than this")
    b.add_argument("--leagues", nargs="*")
    b.add_argument("--include-flagged", action="store_true", help="also fetch region_ok=0 matches")
    b.add_argument("--pause", type=float, default=5, help="seconds between requests (default 5)")
    sub.add_parser("build")
    args = ap.parse_args()
    {"players": cmd_players, "feed": cmd_feed, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()

"""Pro players' solo-queue history from the official Riot API.

Pipeline (scripts/soloq_*.py drive it):
  1. data/soloq/pro_accounts.csv  - which Riot IDs belong to which pro (built by soloq_accounts.py)
  2. account-v1                   - Riot ID -> PUUID              (data/soloq/puuids.csv)
  3. match-v5 ids                 - ranked solo (queue 420) ids    (data/soloq/match_ids/<puuid>.txt)
  4. match-v5 details             - one row per (match, pro)       (data/soloq/matches.csv)

Everything is resumable: each step skips what is already on disk, so an interrupted run
(or an expired 24h development key) just continues next time.

The API key is read from RIOT_API_KEY in the project's .env file (or the environment).
It is never printed or written anywhere else.
"""
from __future__ import annotations

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from pathlib import Path

from .config import ROOT

SOLOQ_DIR = Path(os.environ.get("DRAFTLAB_SOLOQ_DIR", ROOT / "data" / "soloq"))
ACCOUNTS_CSV = SOLOQ_DIR / "pro_accounts.csv"
PUUIDS_CSV = SOLOQ_DIR / "puuids.csv"
MATCH_IDS_DIR = SOLOQ_DIR / "match_ids"
MATCHES_CSV = SOLOQ_DIR / "matches.csv"
FAILED_CSV = SOLOQ_DIR / "failed_matches.csv"

RANKED_SOLO_QUEUE = 420

# Server (as written on stats sites / Leaguepedia) -> Riot regional routing value for match-v5.
SERVER_TO_REGION = {
    "KR": "asia", "JP": "asia", "JP1": "asia",
    "EUW": "europe", "EUW1": "europe", "EUNE": "europe", "EUN1": "europe",
    "TR": "europe", "TR1": "europe", "RU": "europe",
    "NA": "americas", "NA1": "americas", "BR": "americas", "BR1": "americas",
    "LAN": "americas", "LA1": "americas", "LAS": "americas", "LA2": "americas",
    "OCE": "sea", "OC1": "sea", "VN": "sea", "VN2": "sea", "TW": "sea", "TW2": "sea",
    "SG": "sea", "SG2": "sea", "PH": "sea", "TH": "sea", "ME": "europe", "ME1": "europe",
}

MATCH_FIELDS = [
    "match_id", "puuid", "player", "server", "game_start", "patch", "duration_s",
    "champion", "position", "win", "kills", "deaths", "assists",
    "opp_champion", "ally_champions", "enemy_champions",
]


# ---------------------------------------------------------------- key handling

def load_api_key(env_path: Path | None = None) -> str:
    """RIOT_API_KEY from .env at the project root, else from the environment."""
    env_path = env_path or ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            if k.strip() == "RIOT_API_KEY":
                v = v.strip().strip('"').strip("'")
                if v:
                    return v
    v = os.environ.get("RIOT_API_KEY", "").strip()
    if v:
        return v
    raise RuntimeError(
        f"No Riot API key found. Create {env_path} with one line:\n"
        "    RIOT_API_KEY=RGAPI-...\n"
        "(get a development key at https://developer.riotgames.com; it expires every 24 h)."
    )


# ---------------------------------------------------------------- rate limiting

class RateLimiter:
    """Sliding-window limiter. Default = development key: 20 req / 1 s and 100 req / 120 s.

    A small safety margin is kept below each limit. Limits are per routing value on Riot's
    side, but we use one shared limiter: simpler, and the app-wide 100/2min is the binding one.
    """

    def __init__(self, limits=((20, 1.0), (100, 120.0)), margin: int = 1):
        self.limits = [(max(1, n - margin), w) for n, w in limits]
        self.calls: deque[float] = deque()

    def wait(self):
        while True:
            now = time.monotonic()
            longest = max(w for _, w in self.limits)
            while self.calls and now - self.calls[0] > longest:
                self.calls.popleft()
            sleep_for = 0.0
            for n, w in self.limits:
                recent = [t for t in self.calls if now - t < w]
                if len(recent) >= n:
                    sleep_for = max(sleep_for, w - (now - recent[-n]) + 0.05)
            if sleep_for <= 0:
                self.calls.append(now)
                return
            time.sleep(sleep_for)

    def update_from_header(self, header: str | None):
        """Adopt the app limits Riot reports (e.g. '20:1,100:120'), so a production key goes faster."""
        if not header:
            return
        try:
            lim = [(int(a), float(b)) for a, b in (p.split(":") for p in header.split(","))]
        except ValueError:
            return
        new = [(max(1, n - 1), w) for n, w in lim]
        if new != self.limits:
            self.limits = new


class RiotClient:
    def __init__(self, api_key: str, limiter: RateLimiter | None = None, verbose: bool = True):
        self._key = api_key
        self.limiter = limiter or RateLimiter()
        self.verbose = verbose
        self.n_requests = 0

    def get(self, region: str, path: str, params: dict | None = None, max_retries: int = 5):
        """GET https://{region}.api.riotgames.com{path}. Returns parsed JSON, or None on 404."""
        url = f"https://{region}.api.riotgames.com{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        for attempt in range(max_retries):
            self.limiter.wait()
            req = urllib.request.Request(url, headers={"X-Riot-Token": self._key,
                                                       "User-Agent": "LoLDraftLab/0.1"})
            self.n_requests += 1
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    self.limiter.update_from_header(r.headers.get("X-App-Rate-Limit"))
                    return json.loads(r.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return None
                if e.code in (401, 403):
                    raise RuntimeError(
                        f"Riot API refused the key (HTTP {e.code}). Development keys expire after "
                        "24 h: regenerate it at developer.riotgames.com and update .env."
                    ) from None
                if e.code == 429:
                    wait = float(e.headers.get("Retry-After") or 10)
                    if self.verbose:
                        print(f"  429 rate limited ({e.headers.get('X-Rate-Limit-Type')}), waiting {wait:.0f}s")
                    time.sleep(wait + 1)
                    continue
                if 500 <= e.code < 600:
                    time.sleep(2 ** attempt)
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(2 ** attempt)
        raise RuntimeError(f"Giving up after {max_retries} attempts: {path}")

    # -------- endpoints
    def account_by_riot_id(self, game_name: str, tag_line: str, region: str = "asia"):
        p = f"/riot/account/v1/accounts/by-riot-id/{urllib.parse.quote(game_name)}/{urllib.parse.quote(tag_line)}"
        return self.get(region, p)

    def ranked_match_ids(self, puuid: str, region: str, start_time: int | None = None,
                         max_ids: int = 100):
        ids, start = [], 0
        while len(ids) < max_ids:
            count = min(100, max_ids - len(ids))
            params = {"queue": RANKED_SOLO_QUEUE, "type": "ranked", "start": start, "count": count}
            if start_time:
                params["startTime"] = int(start_time)
            batch = self.get(region, f"/lol/match/v5/matches/by-puuid/{puuid}/ids", params) or []
            ids.extend(batch)
            if len(batch) < count:
                break
            start += count
        return ids

    def match(self, match_id: str, region: str):
        return self.get(region, f"/lol/match/v5/matches/{match_id}")


# ---------------------------------------------------------------- parsing helpers

def split_riot_id(riot_id: str) -> tuple[str, str] | None:
    riot_id = (riot_id or "").strip()
    if "#" not in riot_id:
        return None
    name, tag = riot_id.rsplit("#", 1)
    name, tag = name.strip(), tag.strip()
    return (name, tag) if name and tag else None


def region_for(server: str) -> str | None:
    return SERVER_TO_REGION.get((server or "").strip().upper())


def patch_of(game_version: str) -> str:
    """'15.24.734.6211' -> '15.24'. Riot's 2025+ client versions are year-based (25.x shown as 15.x in data)."""
    parts = (game_version or "").split(".")
    return ".".join(parts[:2]) if len(parts) >= 2 else game_version


def summarize_match(m: dict, wanted: dict[str, tuple[str, str]]) -> list[dict]:
    """Rows for the participants of match `m` whose puuid is in `wanted` (puuid -> (player, server))."""
    info = m.get("info") or {}
    parts = info.get("participants") or []
    rows = []
    for p in parts:
        if p.get("puuid") not in wanted:
            continue
        player, server = wanted[p["puuid"]]
        team = p.get("teamId")
        pos = p.get("teamPosition") or p.get("individualPosition") or ""
        allies = [q.get("championName", "") for q in parts if q.get("teamId") == team and q is not p]
        enemies = [q for q in parts if q.get("teamId") != team]
        opp = next((q.get("championName", "") for q in enemies if pos and q.get("teamPosition") == pos), "")
        rows.append({
            "match_id": m["metadata"]["matchId"],
            "puuid": p["puuid"],
            "player": player,
            "server": server,
            "game_start": time.strftime("%Y-%m-%d %H:%M:%S",
                                        time.gmtime((info.get("gameStartTimestamp") or info.get("gameCreation") or 0) / 1000)),
            "patch": patch_of(info.get("gameVersion", "")),
            "duration_s": info.get("gameDuration"),
            "champion": p.get("championName", ""),
            "position": pos,
            "win": int(bool(p.get("win"))),
            "kills": p.get("kills"),
            "deaths": p.get("deaths"),
            "assists": p.get("assists"),
            "opp_champion": opp,
            "ally_champions": "|".join(allies),
            "enemy_champions": "|".join(q.get("championName", "") for q in enemies),
        })
    return rows


# ---------------------------------------------------------------- matching to Oracle's Elixir

OE_ROLES = ("top", "jng", "mid", "bot", "sup")


def norm_name(s) -> str:
    """Case/accent/punctuation-insensitive key for matching player handles across sources."""
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def oe_players(years=("2025", "2026")):
    """Players who appeared in a training league in the given Oracle's Elixir years:
    player, league (of their latest game), team, role, last_pro_game, oe_games."""
    import pandas as pd
    from .config import RAW_DIR, TRAIN_LEAGUES
    cols = ["league", "date", "position", "playername", "teamname"]
    paths = [p for p in sorted(RAW_DIR.glob("*.csv")) if p.name[:4] in years]
    f = pd.concat([pd.read_csv(p, usecols=cols, low_memory=False) for p in paths])
    f = f[f.position.isin(OE_ROLES) & f.league.isin(TRAIN_LEAGUES) & f.playername.notna()]
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    games = f.groupby("playername").size().rename("oe_games")
    last = f.sort_values("date").groupby("playername").tail(1).set_index("playername")
    out = last[["league", "teamname", "position", "date"]].join(games).reset_index()
    return out.rename(columns={"playername": "player", "teamname": "team", "position": "role",
                               "date": "last_pro_game"})


# ---------------------------------------------------------------- small csv helpers

def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_csv(path: Path, rows: list[dict], fields: list[str]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)

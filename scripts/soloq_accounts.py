"""Build data/soloq/pro_accounts.csv from the cached Leaguepedia pages (soloq_leaguepedia.py).

Keeps players who appear in Oracle's Elixir (2025-2026) in a training league, matched by name.
Leaguepedia's SoloqueueIds is free wiki text, e.g.  '''KR:''' Hide on bush#KR1, 2nd acc<br>'''EUW:''' ...
Entries written before Riot IDs (2023) often have no #tag: for those we use the server's default
migration tag (KR1, EUW, NA1, ...), flagged tag_guessed=1, which is right for most old accounts.

    .venv\\Scripts\\python scripts\\soloq_accounts.py
"""
from __future__ import annotations

import html
import json
import re
import sys
import unicodedata
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from draftlab.config import RAW_DIR, TRAIN_LEAGUES  # noqa: E402
from draftlab.soloq import ACCOUNTS_CSV, SERVER_TO_REGION, SOLOQ_DIR  # noqa: E402

CACHE = SOLOQ_DIR / "raw" / "leaguepedia"
DEFAULT_TAG = {"KR": "KR1", "EUW": "EUW", "EUNE": "EUNE", "NA": "NA1", "BR": "BR1", "TR": "TR1",
               "JP": "JP1", "LAN": "LAN", "LAS": "LAS", "OCE": "OCE", "VN": "VN2", "TW": "TW2", "RU": "RU"}
SERVER_ALIASES = {"KOREA": "KR", "KR1": "KR", "EU": "EUW", "EUW1": "EUW", "EU WEST": "EUW", "EUNE1": "EUNE",
                  "NA1": "NA", "NORTH AMERICA": "NA", "BR1": "BR", "BRAZIL": "BR", "TR1": "TR", "TURKEY": "TR",
                  "JP1": "JP", "JAPAN": "JP", "VN2": "VN", "VIETNAM": "VN", "TW2": "TW", "OC1": "OCE",
                  "LA1": "LAN", "LA2": "LAS", "CN": "CN", "CHINA": "CN"}
ROLE_MAP = {"Top": "top", "Jungle": "jng", "Mid": "mid", "Bot": "bot", "Support": "sup"}


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]", "", s.lower())


def clean_wiki(t: str) -> str:
    t = html.unescape(t or "")
    t = re.sub(r"<br\s*/?>", "\n", t, flags=re.I)
    t = re.sub(r"<[^>]+>", "", t)
    t = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", t)   # [[link|text]] -> text
    t = re.sub(r"\[https?://\S+\s+([^\]]*)\]", r"\1", t)       # [url text] -> text
    return t.replace("'''", "").replace("''", "")


def parse_ids(text: str) -> list[tuple[str, str, int]]:
    """-> [(server, riot_id, tag_guessed)]. Lines look like 'KR: a#KR1, b' or '(KR) a'."""
    out = []
    for line in re.split(r"[\n;]", clean_wiki(text)):
        line = line.strip(" *•-\t")
        if not line:
            continue
        m = re.match(r"^\(?([A-Za-z][A-Za-z ]{0,13}?\d?)\)?\s*[:：]\s*(.+)$", line)
        if not m:
            continue
        server = m.group(1).strip().upper()
        server = SERVER_ALIASES.get(server, server)
        for acc in re.split(r",|\s\|\s", m.group(2)):
            acc = re.sub(r"\(.*?\)", "", acc).strip()
            if not acc or len(acc) > 40:
                continue
            if "#" in acc:
                out.append((server, acc, 0))
            elif server in DEFAULT_TAG:
                out.append((server, f"{acc}#{DEFAULT_TAG[server]}", 1))
            else:
                out.append((server, acc, 1))
    return out


def load_leaguepedia() -> pd.DataFrame:
    rows = []
    for f in sorted(CACHE.glob("players_*.json")):
        for item in json.loads(f.read_text(encoding="utf-8")).get("cargoquery", []):
            rows.append(item["title"])
    return pd.DataFrame(rows)


def oe_players() -> pd.DataFrame:
    cols = ["league", "date", "position", "playername", "teamname"]
    paths = [p for p in sorted(RAW_DIR.glob("*.csv")) if p.name[:4] in ("2025", "2026")]
    f = pd.concat([pd.read_csv(p, usecols=cols, low_memory=False) for p in paths])
    f = f[f.position.isin(ROLE_MAP.values()) & f.league.isin(TRAIN_LEAGUES) & f.playername.notna()]
    f["date"] = pd.to_datetime(f["date"], errors="coerce")
    games = f.groupby("playername").size().rename("oe_games_2025_26")
    last = f.sort_values("date").groupby("playername").tail(1).set_index("playername")
    out = last[["league", "teamname", "position", "date"]].join(games).reset_index()
    return out.rename(columns={"playername": "player", "teamname": "team", "position": "role",
                               "date": "last_pro_game"})


def main():
    lp = load_leaguepedia()
    if lp.empty:
        sys.exit(f"No cached Leaguepedia pages in {CACHE}: run scripts/soloq_leaguepedia.py first.")
    oe = oe_players()
    print(f"Leaguepedia players with listed accounts: {len(lp)}; OE training-league players 2025-26: {len(oe)}")

    lp["k_id"] = lp["ID"].map(norm)
    lp_by_id: dict[str, list] = {}
    for _, r in lp.iterrows():
        lp_by_id.setdefault(r["k_id"], []).append(r)

    rows, unmatched = [], []
    for _, p in oe.iterrows():
        cands = lp_by_id.get(norm(p["player"]), [])
        if len(cands) > 1:   # same handle used by several people: prefer same role, then not retired
            same = [c for c in cands if ROLE_MAP.get(str(c.get("Role", "")).split(",")[0].strip()) == p["role"]]
            cands = same or cands
            active = [c for c in cands if str(c.get("IsRetired", "")) in ("", "0", "No")]
            cands = active or cands
        if not cands:
            unmatched.append(p["player"])
            continue
        c = cands[0]
        for server, rid, guessed in parse_ids(c.get("SoloqueueIds", "")):
            rows.append({
                "player": p["player"], "team": p["team"], "league": p["league"], "role": p["role"],
                "server": server, "riot_id": rid, "tag_guessed": guessed,
                "supported": int(server in SERVER_TO_REGION),
                "lp_page": c.get("Page", ""), "lp_residency": c.get("Residency", ""),
                "last_pro_game": str(p["last_pro_game"])[:10], "oe_games_2025_26": p["oe_games_2025_26"],
                "source": "leaguepedia",
            })
    out = pd.DataFrame(rows).drop_duplicates(["player", "server", "riot_id"])
    ACCOUNTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(ACCOUNTS_CSV, index=False, encoding="utf-8")
    oe["mapped"] = oe["player"].isin(set(out["player"]))
    oe.to_csv(SOLOQ_DIR / "oe_players_mapping_status.csv", index=False, encoding="utf-8")
    print(f"{out['player'].nunique()} pros mapped, {len(out)} accounts "
          f"({int(out['supported'].sum())} on Riot-API servers, {int(out['tag_guessed'].sum())} with guessed tag)")
    print(f"Unmatched OE players: {len(unmatched)}")


if __name__ == "__main__":
    main()

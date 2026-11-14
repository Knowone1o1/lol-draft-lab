"""find_clips(query): moments from the events table, mapped to video time.

Returns clips (game_id, start, end, reason) with a link: a YouTube URL with ?t= for public VODs (no
downloading), or an ffmpeg cut command (or a cut file) for local recordings such as scrims.

query is either a dict of filters or a short text query, e.g.
    {"game_id": "...", "event_type": "objective", "subtype": "dragon", "team": "T1", "before_s": 900}
    "dragons T1 before 15:00"   "kills by Faker"   "towers after 20:00 in <game_id>"
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .. import store
from .frames import cut_clip, cut_command

WORD_TYPES = {
    "kill": ("kill", None), "kills": ("kill", None), "fight": ("kill", None), "fights": ("kill", None),
    "dragon": ("objective", "dragon"), "dragons": ("objective", "dragon"), "drake": ("objective", "dragon"),
    "drakes": ("objective", "dragon"), "baron": ("objective", "baron"), "herald": ("objective", "herald"),
    "grubs": ("objective", "horde"), "elder": ("objective", "elder"), "objective": ("objective", None),
    "objectives": ("objective", None), "tower": ("tower", None), "towers": ("tower", None),
    "plate": ("tower", "plate"), "plates": ("tower", "plate"), "ward": ("ward", None), "wards": ("ward", None),
    "item": ("item", None), "items": ("item", None),
}


@dataclass
class Clip:
    game_id: str
    start: float                 # game time (s)
    end: float
    reason: str
    video_start: float | None = None
    video_end: float | None = None
    link: str | None = None
    cut_cmd: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def _mmss(s):
    s = int(max(s, 0))
    return f"{s // 60}:{s % 60:02d}"


def parse_query(text: str) -> dict:
    q: dict = {}
    t = text.strip()
    m = re.search(r"\bbefore\s+(\d{1,2}):(\d{2})", t)
    if m:
        q["before_s"] = int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"\bafter\s+(\d{1,2}):(\d{2})", t)
    if m:
        q["after_s"] = int(m.group(1)) * 60 + int(m.group(2))
    m = re.search(r"\bby\s+([\w.'-]+)", t)
    if m:
        q["player"] = m.group(1)
    m = re.search(r"\bin\s+([\w.\-]+)", t)
    if m:
        q["game_id"] = m.group(1)
    for w in re.findall(r"[a-z]+", t.lower()):
        if w in WORD_TYPES:
            q["event_type"], sub = WORD_TYPES[w]
            if sub:
                q["subtype"] = sub
    rest = re.sub(r"\b(before|after)\s+\d{1,2}:\d{2}|\bby\s+[\w.'-]+|\bin\s+[\w.\-]+", "", t)
    words = [w for w in re.findall(r"[\w.'-]+", rest) if w.lower() not in WORD_TYPES]
    if words:
        q["team"] = " ".join(words)
    return q


def _link(url: str, video_s: float) -> str:
    if not url:
        return None
    if "youtube.com" in url or "youtu.be" in url:
        return f"{url}{'&' if '?' in url else '?'}t={int(max(video_s, 0))}s"
    if url.startswith("http"):
        return f"{url}#t={int(max(video_s, 0))}"
    return url


def find_clips(con, query, pre: float = 15.0, post: float = 10.0, merge_gap: float = 10.0, limit: int = 20,
               cut_dir=None) -> list[Clip]:
    q = parse_query(query) if isinstance(query, str) else dict(query or {})
    sql = ["SELECT e.*, g.blue_team, g.red_team FROM events e JOIN games g USING (game_id) WHERE 1=1"]
    args = []
    for col, key in (("e.game_id", "game_id"), ("e.event_type", "event_type")):
        if q.get(key):
            sql.append(f"AND {col} = ?")
            args.append(q[key])
    if q.get("series_id"):
        sql.append("AND g.series_id = ?")
        args.append(q["series_id"])
    if q.get("subtype"):
        sql.append("AND lower(e.subtype) LIKE ?")
        args.append(f"%{q['subtype'].lower()}%")
    if q.get("player"):
        sql.append("AND lower(e.player) = lower(?)")
        args.append(q["player"])
    if q.get("team"):  # team name -> the side it played on in that game; or 'Blue'/'Red' directly
        sql.append("AND ((lower(?) IN ('blue', 'red') AND lower(e.team) = lower(?)) OR "
                   "(e.team = 'Blue' AND lower(g.blue_team) = lower(?)) OR (e.team = 'Red' AND lower(g.red_team) = lower(?)))")
        args += [q["team"]] * 4
    if q.get("before_s") is not None:
        sql.append("AND e.game_time_s <= ?")
        args.append(q["before_s"])
    if q.get("after_s") is not None:
        sql.append("AND e.game_time_s >= ?")
        args.append(q["after_s"])
    sql.append("ORDER BY e.game_id, e.game_time_s")
    rows = con.execute(" ".join(sql), args).fetchdf()
    clips: list[Clip] = []
    for gid, ev in rows.groupby("game_id", sort=False):
        windows = []
        for r in ev.itertuples():
            s, e = r.game_time_s - pre, r.game_time_s + post
            label = f"{r.subtype or r.event_type}" + (f" ({r.blue_team if r.team == 'Blue' else r.red_team})" if r.team else "")
            if windows and s <= windows[-1][1] + merge_gap:
                windows[-1][1] = max(windows[-1][1], e)
                windows[-1][2].append(label)
            else:
                windows.append([s, e, [label]])
        vod = con.execute("SELECT video_url FROM vod_sync WHERE game_id = ?", [gid]).fetchone()
        for s, e, labels in windows:
            counts = {}
            for lb in labels:
                counts[lb] = counts.get(lb, 0) + 1
            reason = ", ".join(f"{n}x {lb}" if n > 1 else lb for lb, n in counts.items()) + f" at {_mmss(s + pre)}"
            c = Clip(gid, max(s, 0.0), e, reason)
            if vod:
                c.video_start = store.video_time(con, gid, c.start)
                c.video_end = store.video_time(con, gid, c.end)
                url = vod[0]
                c.link = _link(url, c.video_start)
                if url and not url.startswith("http") and Path(url).exists():
                    out = Path(cut_dir or Path(url).parent / "clips") / f"{gid}_{int(c.start)}-{int(c.end)}.mp4"
                    c.cut_cmd = cut_command(url, c.video_start, c.video_end, out)
            else:
                c.reason += " (no VOD synced for this game)"
            clips.append(c)
            if len(clips) >= limit:
                return clips
    return clips


def clip_at(con, game_id: str, game_time_s: float, reason: str, pre: float = 20.0, post: float = 20.0) -> Clip:
    """A clip around one moment (used by deviation flags in Phase 3)."""
    c = Clip(game_id, max(game_time_s - pre, 0.0), game_time_s + post, reason)
    vod = con.execute("SELECT video_url FROM vod_sync WHERE game_id = ?", [game_id]).fetchone()
    if vod:
        c.video_start = store.video_time(con, game_id, c.start)
        c.video_end = store.video_time(con, game_id, c.end)
        c.link = _link(vod[0], c.video_start)
    return c


def cut(clip: Clip) -> Path | None:
    """Run the ffmpeg cut for a clip from a local recording."""
    if not clip.cut_cmd:
        return None
    src = clip.cut_cmd[clip.cut_cmd.index("-i") + 1]
    return cut_clip(src, clip.video_start, clip.video_end, clip.cut_cmd[-1])

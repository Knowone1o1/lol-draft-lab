"""Build the DuckDB data layer (data/draftlab.duckdb) from the configured sources.

    .venv\\Scripts\\python scripts\\ingest.py                         # Oracle's Elixir -> series, games, drafts, players
    .venv\\Scripts\\python scripts\\ingest.py --timelines data/timelines   # + events from Riot-format timeline JSONs
    .venv\\Scripts\\python scripts\\ingest.py --vod data/vods.csv          # + VOD sync points (game_id, video_url, game_time_s, video_time_s)

GRID events are used automatically once GRID_API_KEY is set and draftlab/sources/grid.py is implemented.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab import store  # noqa: E402
from draftlab.sources import GridSource, OraclesElixirSource, RiotTimelineSource, SourceNotConfigured  # noqa: E402


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(store.DB_PATH))
    ap.add_argument("--timelines", help="folder of <game_id>.json Riot match-v5 timelines")
    ap.add_argument("--vod", help="CSV with columns game_id, video_url, game_time_s, video_time_s (one row per sync point)")
    ap.add_argument("--skip-matches", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    with store.connect(args.db) as con:
        if not args.skip_matches:
            print("Matches from Oracle's Elixir:", store.ingest_matches(con, OraclesElixirSource()))
        if args.timelines:
            print("Events from timelines:", store.ingest_events(con, RiotTimelineSource(args.timelines)))
        try:
            g = GridSource()
            print("Events from GRID:", store.ingest_events(con, g))
        except SourceNotConfigured as e:
            print("GRID: skipped -", e)
        except NotImplementedError as e:
            print("GRID: key set but adapter not implemented -", e)
        if args.vod:
            rows = {}
            with open(args.vod, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f):
                    rows.setdefault((r["game_id"], r["video_url"]), []).append((float(r["game_time_s"]), float(r["video_time_s"])))
            for (gid, url), pts in rows.items():
                store.set_vod(con, gid, url, pts, method="csv", source=Path(args.vod).name)
            print(f"VOD sync: {len(rows)} games")
        counts = {t: con.execute(f"SELECT count(*) FROM {t}").fetchone()[0]
                  for t in ("series", "games", "draft_actions", "game_players", "events", "vod_sync", "vod_sync_points")}
    print(f"{args.db}: {counts} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

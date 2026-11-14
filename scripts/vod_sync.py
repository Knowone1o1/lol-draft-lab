"""VOD sync and clips from the command line.

  preview    save one frame (with a 10% grid) to find where the timer is:
             vod_sync.py preview --video scrim.mp4 --at 600 --out frame.png
  calibrate  teach the timer reader the font from one known moment (no pause in the next 40 s):
             vod_sync.py calibrate --video scrim.mp4 --crop 0.47,0.0,0.06,0.04 --video-time 312 --game-time 0:41
  sync       read the timer every N seconds, store sync points for a game:
             vod_sync.py sync --video scrim.mp4 --crop ... --game-id G1 [--every 30]
  manual     store hand-made sync points (e.g. for a public YouTube VOD, which is not downloaded):
             vod_sync.py manual --game-id G1 --url "https://youtu.be/..." --point 0:00=1:02:15 --point 20:00=1:24:40
  clips      find clips and print links, or cut them from a local recording:
             vod_sync.py clips "dragons T1 before 15:00" [--cut]

Templates for the timer font are saved per broadcast style in data/vod/templates/<name>.json.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from draftlab import store  # noqa: E402
from draftlab.config import ROOT  # noqa: E402
from draftlab.vod import TemplateOCR, TesseractOCR, calibrate_template_ocr, cut, find_clips, parse_time, sync_video  # noqa: E402
from draftlab.vod.frames import grab_frame  # noqa: E402

TEMPLATES = ROOT / "data" / "vod" / "templates"


def hms(text: str) -> float:
    parts = [float(p) for p in text.split(":")]
    s = 0.0
    for p in parts:
        s = s * 60 + p
    return s


def crop_arg(text):
    return tuple(float(x) for x in text.split(",")) if text else None


def load_ocr(name):
    try:
        return TesseractOCR() if name == "tesseract" else TemplateOCR.load(TEMPLATES / f"{name}.json")
    except FileNotFoundError:
        sys.exit(f"No timer templates '{name}' yet: run the calibrate step first.")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("preview"); p.add_argument("--video", required=True); p.add_argument("--at", default="600"); p.add_argument("--out", default="frame.png")
    p = sub.add_parser("calibrate"); p.add_argument("--video", required=True); p.add_argument("--crop", required=True)
    p.add_argument("--video-time", required=True); p.add_argument("--game-time", required=True); p.add_argument("--name", default="default")
    p = sub.add_parser("sync"); p.add_argument("--video", required=True); p.add_argument("--crop", required=True)
    p.add_argument("--game-id", required=True); p.add_argument("--every", type=float, default=30); p.add_argument("--name", default="default")
    p.add_argument("--url", help="link to store instead of the local path (e.g. the same VOD on YouTube)")
    p = sub.add_parser("manual"); p.add_argument("--game-id", required=True); p.add_argument("--url", required=True)
    p.add_argument("--point", action="append", required=True, help="GAME=VIDEO, e.g. 0:00=1:02:15")
    p = sub.add_parser("clips"); p.add_argument("query"); p.add_argument("--cut", action="store_true"); p.add_argument("--limit", type=int, default=10)
    a = ap.parse_args()

    if a.cmd == "preview":
        from PIL import ImageDraw
        img = grab_frame(a.video, hms(a.at))
        d = ImageDraw.Draw(img)
        for i in range(1, 10):
            d.line([(img.width * i / 10, 0), (img.width * i / 10, img.height)], fill=(255, 0, 0))
            d.line([(0, img.height * i / 10), (img.width, img.height * i / 10)], fill=(255, 0, 0))
        img.save(a.out)
        print(f"Saved {a.out} ({img.width}x{img.height}). Red lines every 10%: give --crop as x,y,w,h fractions around the timer.")
    elif a.cmd == "calibrate":
        ocr = TemplateOCR()
        crop, g0 = crop_arg(a.crop), parse_time(a.game_time)
        if g0 is None:
            sys.exit("--game-time must look like 0:41")
        n = calibrate_template_ocr(ocr, lambda v: grab_frame(a.video, v, crop), hms(a.video_time), g0)
        TEMPLATES.mkdir(parents=True, exist_ok=True)
        ocr.save(TEMPLATES / f"{a.name}.json")
        print(f"Learned from {n}/40 frames; characters: {ocr.known_chars}. Saved templates '{a.name}'.")
        if n < 20:
            print("Few frames matched: check the crop (only the timer, a little margin) and the anchor times.")
    elif a.cmd == "sync":
        ocr = load_ocr(a.name)
        pts, readings = sync_video(a.video, crop_arg(a.crop), ocr, every=a.every)
        ok = sum(1 for _, g in readings if g is not None)
        with store.connect() as con:
            store.set_vod(con, a.game_id, a.url or str(Path(a.video).resolve()), pts, method="ocr", source=ocr.name)
        print(f"Read the timer on {ok}/{len(readings)} frames; stored {len(pts)} sync points for {a.game_id}.")
    elif a.cmd == "manual":
        pts = []
        for p in a.point:
            g, v = p.split("=")
            pts.append((hms(g), hms(v)))
        with store.connect() as con:
            store.set_vod(con, a.game_id, a.url, sorted(pts), method="manual")
        print(f"Stored {len(pts)} sync points for {a.game_id}.")
    elif a.cmd == "clips":
        with store.connect(read_only=True) as con:
            clips = find_clips(con, a.query, limit=a.limit)
        if not clips:
            print("No matching events (the events table fills from scrim timelines or GRID data).")
        for c in clips:
            print(f"{c.game_id} {int(c.start)//60}:{int(c.start)%60:02d}-{int(c.end)//60}:{int(c.end)%60:02d}  {c.reason}  {c.link or ''}")
            if a.cut and c.cut_cmd:
                print("   cut ->", cut(c))


if __name__ == "__main__":
    main()

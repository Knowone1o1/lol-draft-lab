"""Phase 2: timer OCR, sync with pauses and misreads, clip finder, and an end-to-end run on a generated video."""
import random
import subprocess

import pytest
from PIL import Image, ImageDraw, ImageFont

from draftlab import store
from draftlab.vod import (TemplateOCR, calibrate_template_ocr, find_clips, fit_sync_points, format_time,
                          parse_query, parse_time, sync_video)
from draftlab.vod.frames import ffmpeg_exe, grab_frame

FONT = ImageFont.load_default(size=22)


def timer_frame(game_s, noise=0, size=(120, 36), seed=0):
    """A frame that looks like a game timer strip: light digits on a dark background."""
    img = Image.new("RGB", size, (12, 16, 24))
    d = ImageDraw.Draw(img)
    d.text((14, 5), format_time(game_s), font=FONT, fill=(235, 235, 220))
    if noise:
        rnd = random.Random(seed)
        px = img.load()
        for _ in range(noise):
            x, y = rnd.randrange(size[0]), rnd.randrange(size[1])
            px[x, y] = (40, 40, 50)
    return img


def test_parse_time():
    assert parse_time("14:37") == 877 and parse_time("00:05") == 5
    assert parse_time("14:7") is None and parse_time("12:75") is None and parse_time(None) is None


def test_template_ocr_learns_from_one_anchor_and_reads_new_times():
    ocr = TemplateOCR()
    learned = calibrate_template_ocr(ocr, lambda v: timer_frame(int(v)), anchor_video_s=41, anchor_game_s=41, seconds=40)
    assert learned == 40
    for t in (123, 754, 1999, 2468, 3599, 609):
        text, conf = ocr.read(timer_frame(t, noise=8, seed=t))
        assert text == format_time(t), (t, text, conf)


def test_template_ocr_save_load(tmp_path):
    ocr = TemplateOCR()
    calibrate_template_ocr(ocr, lambda v: timer_frame(int(v)), 0, 0, seconds=20)
    ocr.save(tmp_path / "t.json")
    again = TemplateOCR.load(tmp_path / "t.json")
    assert again.read(timer_frame(1318))[0] == "21:58"


def test_fit_sync_points_handles_pause_and_misreads(db):
    # video = game + 100 s until game 600; a 60 s pause; then video = game + 160 s
    readings, rnd = [], random.Random(1)
    for v in range(100, 1400, 30):
        g = v - 100 if v - 100 <= 600 else (600 if v - 160 < 600 else v - 160)
        if rnd.random() < 0.15:
            g = rnd.randrange(0, 2000)          # misread
        if rnd.random() < 0.1:
            g = None                            # unreadable frame
        readings.append((v, g))
    pts = fit_sync_points(readings)
    store.set_vod(db, "G", "video.mp4", pts, method="ocr")
    assert abs(store.video_time(db, "G", 300) - 400) <= 2
    assert abs(store.video_time(db, "G", 1000) - 1160) <= 2
    gs = [g for g, _ in pts]
    assert gs == sorted(gs)                      # game time only moves forward


def test_parse_query():
    q = parse_query("dragons T1 before 15:00")
    assert q["event_type"] == "objective" and q["subtype"] == "dragon" and q["team"] == "T1" and q["before_s"] == 900
    q = parse_query("kills by Faker after 20:00")
    assert q["event_type"] == "kill" and q["player"] == "Faker" and q["after_s"] == 1200 and "team" not in q


@pytest.fixture()
def game_with_events(db):
    db.execute("INSERT INTO games VALUES ('G1', 'S1', 1, now(), 'LCK', '16.20', 'T1', 'Gen.G', 1, 'test')")
    rows = [("G1", 400, "objective", "fire_dragon", "Red", None), ("G1", 405, "kill", "kill", "Blue", "Faker"),
            ("G1", 412, "kill", "kill", "Red", "Chovy"), ("G1", 1300, "objective", "baron_nashor", "Blue", None),
            ("G1", 950, "objective", "earth_dragon", "Blue", None)]
    for gid, t, et, sub, team, pl in rows:
        db.execute("INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, NULL, 'test')", [gid, t, et, sub, team, pl])
    return db


def test_find_clips_youtube_links_and_merging(game_with_events):
    db = game_with_events
    store.set_vod(db, "G1", "https://www.youtube.com/watch?v=abc", [(0, 300), (1000, 1300)])
    clips = find_clips(db, "dragons Gen.G")             # Gen.G was red side: only the fire dragon
    assert len(clips) == 1 and "fire_dragon (Gen.G)" in clips[0].reason
    assert clips[0].link == "https://www.youtube.com/watch?v=abc&t=685s"   # 400 - 15 s lead-in + 300 offset
    fight = find_clips(db, {"game_id": "G1", "before_s": 600})            # dragon + 2 kills within seconds -> one clip
    assert len(fight) == 1 and "2x kill" not in fight[0].reason and fight[0].end >= 422
    no_vod = find_clips(db, {"event_type": "objective", "subtype": "baron"})
    assert no_vod[0].link is not None


def test_find_clips_without_vod(game_with_events):
    clips = find_clips(game_with_events, "baron")
    assert len(clips) == 1 and "no VOD synced" in clips[0].reason and clips[0].link is None


def test_end_to_end_generated_video(tmp_path, db):
    """Render a 6-minute 'broadcast' with a 30 s pause, encode it with ffmpeg, then calibrate, sync and cut."""
    try:
        exe = ffmpeg_exe()
    except RuntimeError:
        pytest.skip("ffmpeg not available")
    frames = tmp_path / "f"
    frames.mkdir()
    offset, pause_at, pause_len = 20, 200, 30           # timer starts at video 20 s; pause at game 200 s
    for v in range(0, 360):
        g = 0 if v < offset else (v - offset if v - offset <= pause_at else (pause_at if v - offset < pause_at + pause_len else v - offset - pause_len))
        big = Image.new("RGB", (320, 180), (30, 60, 40))
        big.paste(timer_frame(g), (100, 10))
        big.save(frames / f"f{v:04d}.png")
    video = tmp_path / "scrim.avi"
    subprocess.run([exe, "-hide_banner", "-loglevel", "error", "-y", "-framerate", "1", "-i", str(frames / "f%04d.png"),
                    "-c:v", "mpeg4", "-q:v", "2", "-r", "1", str(video)], check=True)
    crop = (100 / 320, 10 / 180, 120 / 320, 36 / 180)
    ocr = TemplateOCR()
    learned = calibrate_template_ocr(ocr, lambda v: grab_frame(video, v, crop), anchor_video_s=30, anchor_game_s=10, seconds=30)
    assert learned >= 25
    pts, readings = sync_video(video, crop, ocr, every=15, start=0.0)
    store.set_vod(db, "G1", str(video), pts, method="ocr")
    assert abs(store.video_time(db, "G1", 100) - 120) <= 2               # before the pause: +20 s
    assert abs(store.video_time(db, "G1", 260) - 310) <= 2               # after the pause: +50 s
    db.execute("INSERT INTO games VALUES ('G1', 'S1', 1, now(), 'SCRIM', NULL, 'Us', 'Them', 1, 'test')")
    db.execute("INSERT INTO events VALUES ('G1', 260, 'kill', 'kill', 'Blue', 'Faker', NULL, NULL, NULL, 'test')")
    clip = find_clips(db, {"game_id": "G1"}, cut_dir=tmp_path / "clips")[0]
    assert clip.cut_cmd, "local recordings get an ffmpeg cut"
    from draftlab.vod import cut
    out = cut(clip)
    assert out.exists() and out.stat().st_size > 0

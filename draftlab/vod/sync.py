"""Sync a video to in-game time by reading the timer at intervals.

One fixed offset breaks as soon as the game is paused (the video keeps running, the timer stops), so we
read the timer every `every` seconds of video and keep many (game_time, video_time) points. Between two
points the store interpolates; a pause shows up as the offset (video - game) jumping up between readings.

Misreads are rejected: a reading is kept only if its offset agrees (within `tol` s) with the current
offset, or if two consecutive readings agree on a NEW, larger offset (a pause). Offsets going down would
mean the game clock ran faster than the video, which only a misread (or a replay) produces.
"""
from __future__ import annotations

from typing import Callable

from .ocr import format_time, parse_time


def fit_sync_points(readings: list[tuple[float, int | None]], tol: float = 2.0, min_run: int = 2) -> list[tuple[float, float]]:
    """readings: [(video_time_s, game_time_s or None)] in video order -> [(game_time_s, video_time_s)]."""
    pts: list[tuple[float, float]] = []
    current = None          # accepted offset
    pending: list[tuple[float, int]] = []
    for v, g in readings:
        if g is None:
            continue
        off = v - g
        if current is None:
            pending.append((v, g))
            offs = [pv - pg for pv, pg in pending]
            if len(pending) >= min_run and max(offs) - min(offs) <= tol:
                current = sum(offs) / len(offs)
                pts.extend((pg, pv) for pv, pg in pending)
                pending = []
            elif len(pending) >= min_run:
                pending = pending[-1:]
            continue
        if abs(off - current) <= tol:
            if not pts or g > pts[-1][0]:
                pts.append((g, v))
            pending = []
        elif off > current + tol:
            # possible pause: accept once `min_run` consecutive readings agree on the new offset
            pending.append((v, g))
            offs = [pv - pg for pv, pg in pending]
            if max(offs) - min(offs) > tol:
                pending = pending[-1:]
            elif len(pending) >= min_run:
                current = sum(offs) / len(offs)
                pts.extend((pg, pv) for pv, pg in pending if not pts or pg > pts[-1][0])
                pending = []
        else:
            pending = []    # offset went backwards: misread or replay, ignore
    return pts


def read_timer_series(video_times: list[float], read_at: Callable[[float], tuple[str | None, float]],
                      min_conf: float = 0.5) -> list[tuple[float, int | None]]:
    out = []
    for v in video_times:
        text, conf = read_at(v)
        out.append((v, parse_time(text) if text and conf >= min_conf else None))
    return out


def calibrate_template_ocr(ocr, grab: Callable[[float], object], anchor_video_s: float, anchor_game_s: int,
                           seconds: int = 40) -> int:
    """Teach TemplateOCR the timer font from one known moment: for the next `seconds` seconds the timer is
    anchor_game_s + k (assumes no pause in that window). Returns how many frames were learned."""
    n = 0
    for k in range(seconds):
        n += bool(ocr.learn(grab(anchor_video_s + k), format_time(anchor_game_s + k)))
    return n


def sync_video(path, crop, ocr, every: float = 30.0, start: float = 0.0, end: float | None = None,
               tol: float = 2.0) -> tuple[list[tuple[float, float]], list[tuple[float, int | None]]]:
    """Sample the timer every `every` seconds and return (sync points, raw readings)."""
    from .frames import duration_s, grab_frame
    end = end or duration_s(path)
    times = []
    t = start
    while t < end:
        times.append(t)
        t += every
    readings = read_timer_series(times, lambda v: ocr.read(grab_frame(path, v, crop)))
    return fit_sync_points(readings, tol=tol), readings

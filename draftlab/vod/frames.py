"""Read single frames from a video and cut clips, with ffmpeg (the copy bundled by imageio-ffmpeg, or one
on PATH). Works on local files only: downloading broadcast VODs from YouTube is against YouTube's terms,
so public VODs get timestamp links instead of cuts (see clips.py)."""
from __future__ import annotations

import io
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def ffmpeg_exe() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("ffmpeg not found: pip install imageio-ffmpeg, or install ffmpeg and add it to PATH")
        return exe


def duration_s(path) -> float:
    """Video length in seconds (parsed from ffmpeg's header output)."""
    out = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True).stderr
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", out)
    if not m:
        raise RuntimeError(f"could not read the duration of {path}")
    h, mnt, s = m.groups()
    return int(h) * 3600 + int(mnt) * 60 + float(s)


def grab_frame(path, t: float, crop: tuple[float, float, float, float] | None = None) -> Image.Image:
    """The first frame at or after video time t (seconds): within one frame (1/30 s at 30 fps).
    crop = (x, y, w, h) as fractions of the frame (0-1)."""
    vf = []
    if crop:
        x, y, w, h = crop
        vf = ["-vf", f"crop=iw*{w}:ih*{h}:iw*{x}:ih*{y}"]
    # Fast seek to 5 s before the target (keyframe-based), then an exact seek for the rest: frame-accurate on
    # any container without decoding a multi-hour VOD from the start.
    t = max(t, 0.0)
    coarse = max(t - 5.0, 0.0)
    cmd = [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{coarse:.3f}", "-i", str(path),
           "-ss", f"{t - coarse:.3f}", "-frames:v", "1", *vf, "-f", "image2pipe", "-vcodec", "png", "-"]
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode != 0 or not res.stdout:
        raise RuntimeError(f"no frame at {t:.1f}s: {res.stderr.decode(errors='ignore')[:200]}")
    return Image.open(io.BytesIO(res.stdout)).convert("RGB")


def cut_command(path, start: float, end: float, out) -> list[str]:
    """ffmpeg command that cuts [start, end] of a local video into `out` (re-encoded, frame-accurate)."""
    return [ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y", "-ss", f"{max(start, 0):.2f}", "-to", f"{end:.2f}",
            "-i", str(path), "-c:v", "mpeg4", "-q:v", "4", "-an", str(out)]


def cut_clip(path, start: float, end: float, out) -> Path:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    res = subprocess.run(cut_command(path, start, end, out), capture_output=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.decode(errors="ignore")[:300])
    return Path(out)

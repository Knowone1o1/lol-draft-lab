"""Read the in-game timer (e.g. "14:37") from a cropped frame.

TemplateOCR needs no installs: it learns what each character looks like from frames whose time is known
(calibrate from one anchor: "at video 0:05:12 the timer showed 00:41"), then matches glyphs by correlation.
The game's timer font never changes within a broadcast, so a few dozen labelled frames cover 0-9 and ':'.
TesseractOCR is used instead when the `tesseract` program is installed.
"""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

GLYPH = (12, 18)  # normalised glyph size (w, h)
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


def parse_time(text: str | None) -> int | None:
    """'14:37' -> 877 seconds; None for anything that isn't a valid mm:ss."""
    if not text:
        return None
    m = TIME_RE.match(text.strip())
    if not m or int(m.group(2)) >= 60:
        return None
    return int(m.group(1)) * 60 + int(m.group(2))


def format_time(s: int) -> str:
    return f"{s // 60:02d}:{s % 60:02d}"


def binarize(img: Image.Image) -> np.ndarray:
    """Text pixels = 1. Otsu threshold on grayscale; the text is the minority class."""
    g = np.asarray(img.convert("L"), dtype=float)
    hist, edges = np.histogram(g, bins=64, range=(0, 255))
    w = np.cumsum(hist)
    mu = np.cumsum(hist * edges[:-1])
    total, mu_t = w[-1], mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w - mu * total) ** 2 / (w * (total - w))
    t = edges[int(np.nanargmax(between))]
    b = g > t
    if b.mean() > 0.5:
        b = ~b
    return b.astype(np.uint8)


def segment(b: np.ndarray, min_pixels: int = 2) -> list[np.ndarray]:
    """Split a binary strip into glyphs at empty columns; trim each glyph to its rows."""
    cols = b.sum(0) > 0
    glyphs, start = [], None
    for x, on in enumerate(list(cols) + [False]):
        if on and start is None:
            start = x
        elif not on and start is not None:
            g = b[:, start:x]
            rows = np.nonzero(g.sum(1))[0]
            g = g[rows[0]:rows[-1] + 1]
            if g.sum() >= min_pixels:
                glyphs.append(g)
            start = None
    return glyphs


def _norm(g: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = g.shape
    img = Image.fromarray((g * 255).astype(np.uint8)).resize(GLYPH, Image.BILINEAR)
    a = np.asarray(img, dtype=float).ravel()
    a = (a - a.mean()) / (a.std() + 1e-6)
    return a, w / max(h, 1)


class TemplateOCR:
    name = "template"

    def __init__(self, templates: dict | None = None):
        self.t: dict[str, list[tuple[np.ndarray, float]]] = templates or {}

    # ---- learning
    def learn(self, img: Image.Image, text: str) -> bool:
        """Add glyph templates from a frame whose timer text is known. False if segmentation disagrees."""
        gl = segment(binarize(img))
        if len(gl) != len(text):
            return False
        for ch, g in zip(text, gl):
            lst = self.t.setdefault(ch, [])
            if len(lst) < 12:
                lst.append(_norm(g))
        return True

    @property
    def known_chars(self) -> str:
        return "".join(sorted(self.t))

    # ---- reading
    def read(self, img: Image.Image) -> tuple[str | None, float]:
        """(text, confidence). Confidence = the weakest glyph's correlation (1 = perfect)."""
        if not self.t:
            raise RuntimeError("TemplateOCR has no templates: calibrate it first")
        out, conf = [], 1.0
        for g in segment(binarize(img)):
            a, asp = _norm(g)
            best, best_s = None, -9.0
            for ch, lst in self.t.items():
                for ta, tasp in lst:
                    s = float(np.dot(a, ta) / a.size) - 2.0 * abs(asp - tasp)
                    if s > best_s:
                        best, best_s = ch, s
            out.append(best)
            conf = min(conf, best_s)
        text = "".join(c for c in out if c)
        return (text if parse_time(text) is not None else None), conf

    # ---- persistence
    def save(self, path):
        data = {ch: [{"a": a.round(4).tolist(), "aspect": asp} for a, asp in lst] for ch, lst in self.t.items()}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path):
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({ch: [(np.array(x["a"]), x["aspect"]) for x in lst] for ch, lst in raw.items()})


class TesseractOCR:
    """Uses the `tesseract` program if installed (single text line, digits and ':' only)."""
    name = "tesseract"

    def __init__(self, exe: str | None = None):
        self.exe = exe or shutil.which("tesseract")
        if not self.exe:
            raise RuntimeError("tesseract is not installed")

    def read(self, img: Image.Image) -> tuple[str | None, float]:
        big = img.convert("L").resize((img.width * 3, img.height * 3))
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.png"
            big.save(p)
            res = subprocess.run([self.exe, str(p), "stdout", "--psm", "7", "-c", "tessedit_char_whitelist=0123456789:"],
                                 capture_output=True, text=True)
        text = res.stdout.strip().replace(" ", "")
        return (text if parse_time(text) is not None else None), (1.0 if parse_time(text) is not None else 0.0)

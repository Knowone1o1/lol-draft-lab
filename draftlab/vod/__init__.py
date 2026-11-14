"""Phase 2: VOD sync (timer OCR at intervals) and clip finder."""
from .clips import Clip, clip_at, cut, find_clips, parse_query
from .ocr import TemplateOCR, TesseractOCR, format_time, parse_time
from .sync import calibrate_template_ocr, fit_sync_points, sync_video

__all__ = ["Clip", "TemplateOCR", "TesseractOCR", "calibrate_template_ocr", "clip_at", "cut", "find_clips",
           "fit_sync_points", "format_time", "parse_query", "parse_time", "sync_video"]

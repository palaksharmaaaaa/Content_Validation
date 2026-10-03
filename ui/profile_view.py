"""ui.profile_view: adds the display-oriented blocks the Streamlit renderers read to a modality profile dict."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from audio_detector import AudioProfiler
from image_detector import ImageProfiler
from video_detector import VideoProfiler

_MIME_DEFAULT = {"image": "png", "video": "mp4", "audio": "wav"}
_FORMAT_KEY = {"image": "format", "video": "container_format", "audio": "format"}


def _spec_block(modality: str, res: Dict[str, Any]) -> Dict[str, Any]:
    if modality == "image":
        return {"pixel_specifications": {k: res.get(k) for k in ("width", "height", "aspect_ratio", "channels", "pixel_entropy")}}
    if modality == "video":
        return {"stream_specifications": {k: res.get(k) for k in ("duration_seconds", "fps", "total_frames", "width", "height", "codec")}}
    return {"signal_specifications": {
        k: res.get(k) for k in ("sample_rate", "duration_seconds", "rms_energy", "crest_factor", "dynamic_range_db", "is_clipped")
    }}


def add_ui_profile_blocks(modality: str, res: Dict[str, Any]) -> Dict[str, Any]:
    """Mutates and returns a valid profile with ``success``, ``file_identity`` and the per-modality spec block."""
    if not res.get("valid"):
        return res
    container = _FORMAT_KEY[modality]
    res["success"] = True
    res["file_identity"] = {
        "filename": res.get("filename"),
        "container_format": res.get(container),
        "size_kb": (res.get("file_size_bytes", 0) / 1024.0),
        "mime_type": f"{modality}/{res.get(container, _MIME_DEFAULT[modality]).lower()}",
        "sha256": res.get("sha256"),
    }
    res.update(_spec_block(modality, res))
    return res


def profile_media(file_path: str | Path, modality: str = "auto", source: str = "User Upload") -> Dict[str, Any]:
    p = Path(file_path)
    suffix = p.suffix.lower()
    if modality == "image" or (modality == "auto" and suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff")):
        return add_ui_profile_blocks("image", ImageProfiler().profile_image(file_path, source=source))
    if modality == "video" or (modality == "auto" and suffix in (".mp4", ".mov", ".avi", ".mkv", ".webm")):
        return add_ui_profile_blocks("video", VideoProfiler().profile_video(file_path))
    if modality == "audio" or (modality == "auto" and suffix in (".wav", ".mp3", ".aac", ".flac", ".ogg", ".m4a")):
        return add_ui_profile_blocks("audio", AudioProfiler().profile_audio(file_path))
    return {"success": False, "error": f"Unknown format: {suffix}"}

"""ui.profile_view: adds the display-oriented blocks the Streamlit renderers read to a modality profile dict."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from audio_detector import AudioProfiler
from image_detector import ImageProfiler
from video_detector import VideoProfiler

_MIME_DEFAULT = {"image": "png", "video": "mp4", "audio": "wav"}
_FORMAT_KEY = {"image": "format", "video": "container_format", "audio": "format"}


def add_ui_profile_blocks(modality: str, res: Dict[str, Any]) -> Dict[str, Any]:
    """Mutates and returns a valid profile with ``success`` and ``file_identity`` (the measurements stay in the profiler's own blocks)."""
    if not res.get("valid"):
        return res
    container = _FORMAT_KEY[modality]
    res["success"] = True
    res["file_identity"] = {
        "filename": res.get("filename"),
        "container_format": res.get(container),
        "size_kb": (res.get("file_size_bytes", 0) / 1024.0),
        "mime_type": f"{modality}/{(res.get(container) or _MIME_DEFAULT[modality]).lower()}",
        "sha256": res.get("sha256"),
    }
    return res


def modality_of_extension(suffix: str) -> Optional[str]:
    """The modality whose package supports this file extension (the packages' own SUPPORTED_EXTENSIONS, so they cannot drift)."""
    import audio_detector.config as audio_cfg
    import image_detector.config as image_cfg
    import video_detector.config as video_cfg

    for modality, cfg in (("image", image_cfg), ("video", video_cfg), ("audio", audio_cfg)):
        if suffix in cfg.SUPPORTED_EXTENSIONS:
            return modality
    return None


def profile_media(file_path: str | Path, modality: str = "auto", source: str = "User Upload") -> Dict[str, Any]:
    """Profile a file with its modality's profiler and add the display blocks the result page shows."""
    p = Path(file_path)
    if modality == "auto":
        modality = modality_of_extension(p.suffix.lower()) or "unknown"
    if modality == "image":
        return add_ui_profile_blocks("image", ImageProfiler().profile_image(file_path, source=source))
    if modality == "video":
        return add_ui_profile_blocks("video", VideoProfiler().profile_video(file_path))
    if modality == "audio":
        return add_ui_profile_blocks("audio", AudioProfiler().profile_audio(file_path))
    return {"success": False, "error": f"Unknown format: {p.suffix.lower()}"}

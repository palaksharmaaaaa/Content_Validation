"""ui.profile_view: adds the display-oriented blocks the Streamlit renderers read to a modality profile dict."""
from __future__ import annotations

from typing import Any, Dict

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

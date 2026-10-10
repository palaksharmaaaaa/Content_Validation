"""
video_detector.attribution: Generative AI Video Model Attribution & Fingerprinting Engine.
Profiles video characteristics against international commercial and open-source video models:
- ByteDance Seedance / Jimeng AI
- Kuaishou Kling AI
- OpenAI Sora
- Runway Gen-3 Alpha
- Google Veo
- Luma Dream Machine
- Pika Labs
- MiniMax Hailuo AI
- Alibaba Wan / Tongyi Wanxiang
- Tencent Hunyuan Video
- Vidu (Shengshu Technology)
- Pixverse
- Meta Movie Gen
- Adobe Firefly Video
- Amazon Nova Reel
Completely self-contained with zero outside dependencies.

NOTE on scoring confidence per entry: the original eight entries have dedicated motion/
flicker heuristic signals below in addition to vendor-metadata matching. The 2025/2026
additions (Wan, Hunyuan, Vidu, Pixverse, Movie Gen, Firefly Video, Nova Reel) are currently
vendor-metadata-signature-only -- matched when a vendor name/string appears in container
metadata (see provenance.py's vendor_signatures_found), with no dedicated motion/artifact
heuristic yet. Add real calibration for them only once backed by actual sample analysis.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
from core.shared_results import finalize_attribution, unknown_attribution

logger = logging.getLogger("video_detector.attribution")

KNOWN_VIDEO_GENERATORS = {
    "bytedance_seedance": {
        "name": "ByteDance Seedance / Jimeng AI (即梦)",
        "provider": "ByteDance",
        "telltales": ["H.264/HEVC web profile", "ByteDance MP4 atom headers", "smooth cinematic camera glide"],
    },
    "kuaishou_kling": {
        "name": "Kuaishou Kling AI (可灵)",
        "provider": "Kuaishou Technology",
        "telltales": ["Corner logo watermark on standard tier", "characteristic physics micro-jitter"],
    },
    "openai_sora": {
        "name": "OpenAI Sora",
        "provider": "OpenAI",
        "telltales": ["C2PA Content Credentials signature", "animated corner emblem", "DiT spatial consistency"],
    },
    "runway_gen": {
        "name": "Runway Gen-2 / Gen-3 Alpha",
        "provider": "RunwayML",
        "telltales": ["Runway bottom-right brand watermark", "high dynamic camera motion flow"],
    },
    "google_veo": {
        "name": "Google Veo (Veo 2 / 3.1)",
        "provider": "Google DeepMind",
        "telltales": ["SynthID video watermark", "consistent physics and cinematic motion blur"],
    },
    "luma_dream_machine": {
        "name": "Luma Dream Machine",
        "provider": "Luma AI",
        "telltales": ["Rapid motion flow", "rapid zoom and dolly trajectory patterns"],
    },
    "pika": {
        "name": "Pika Labs (Pika 1.0 / 2.0)",
        "provider": "Pika Labs",
        "telltales": ["Pika watermark emblem", "lip-sync audio overlay"],
    },
    "minimax_hailuo": {
        "name": "MiniMax Hailuo AI",
        "provider": "MiniMax",
        "telltales": ["High frame rate motion extrapolation", "cinematic lighting shifts"],
    },
    "alibaba_wan": {
        "name": "Alibaba Wan / Tongyi Wanxiang",
        "provider": "Alibaba",
        "telltales": ["Vendor metadata signature match only (no motion/artifact fingerprint calibrated yet)"],
    },
    "tencent_hunyuan": {
        "name": "Tencent Hunyuan Video",
        "provider": "Tencent",
        "telltales": ["Vendor metadata signature match only (no motion/artifact fingerprint calibrated yet)"],
    },
    "vidu": {
        "name": "Vidu",
        "provider": "Shengshu Technology",
        "telltales": ["Vendor metadata signature match only (no motion/artifact fingerprint calibrated yet)"],
    },
    "pixverse": {
        "name": "Pixverse",
        "provider": "Pixverse",
        "telltales": ["Vendor metadata signature match only (no motion/artifact fingerprint calibrated yet)"],
    },
    "meta_movie_gen": {
        "name": "Meta Movie Gen",
        "provider": "Meta AI",
        "telltales": ["Vendor metadata signature match only (no motion/artifact fingerprint calibrated yet)"],
    },
    "adobe_firefly_video": {
        "name": "Adobe Firefly Video",
        "provider": "Adobe",
        "telltales": ["Vendor metadata signature match only (no motion/artifact fingerprint calibrated yet)"],
    },
    "amazon_nova_reel": {
        "name": "Amazon Nova Reel",
        "provider": "Amazon",
        "telltales": ["Vendor metadata signature match only (no motion/artifact fingerprint calibrated yet)"],
    },
}


# (needles in the lower-cased vendor string, generator key, cue). First matching row wins per signature.
_VENDOR_SIGNATURES = (
    (("bytedance", "jimeng"), "bytedance_seedance", "ByteDance video metadata signature detected"),
    (("kling", "kuaishou"), "kuaishou_kling", "Kling metadata signature detected"),
    (("runway", "gen-2", "gen-3"), "runway_gen", "Runway metadata identifier detected"),
    (("sora", "openai"), "openai_sora", "OpenAI Sora metadata signature detected"),
    (("veo", "deepmind"), "google_veo", "Google Veo metadata signature detected"),
    (("luma", "dream machine"), "luma_dream_machine", "Luma Dream Machine metadata signature detected"),
    (("pika",), "pika", "Pika Labs metadata signature detected"),
    (("hailuo", "minimax"), "minimax_hailuo", "MiniMax Hailuo metadata signature detected"),
    (("tongyi", "wanxiang"), "alibaba_wan", "Alibaba Wan/Tongyi Wanxiang metadata signature detected"),
    (("hunyuan",), "tencent_hunyuan", "Tencent Hunyuan Video metadata signature detected"),
    (("vidu", "shengshu"), "vidu", "Vidu metadata signature detected"),
    (("pixverse",), "pixverse", "Pixverse metadata signature detected"),
    (("movie gen", "moviegen"), "meta_movie_gen", "Meta Movie Gen metadata signature detected"),
    (("firefly",), "adobe_firefly_video", "Adobe Firefly Video metadata signature detected"),
    (("nova reel", "novareel"), "amazon_nova_reel", "Amazon Nova Reel metadata signature detected"),
)
_CHINA = ("bytedance_seedance", "kuaishou_kling", "minimax_hailuo", "alibaba_wan", "tencent_hunyuan", "vidu")
_UNITED_STATES = ("openai_sora", "google_veo", "runway_gen", "luma_dream_machine", "pika",
                  "meta_movie_gen", "adobe_firefly_video", "amazon_nova_reel")


def _region_for(model_key: str) -> str:
    if model_key in _CHINA:
        return "China"
    return "United States" if model_key in _UNITED_STATES else "Global"


def _score_vendor_signatures(provenance_data: Dict[str, Any], scores: Dict[str, float], cues: List[str]) -> bool:
    """Vendor strings found in container metadata (unauthenticated claims: they steer the guess, never the verdict).

    container_atoms only ever holds generic MP4 box type names and can never contain a vendor identifier; vendor
    strings are scanned separately into vendor_signatures_found.
    """
    declared = False
    for sig in provenance_data.get("vendor_signatures_found", []):
        sig_str = str(sig).lower()
        for needles, key, cue in _VENDOR_SIGNATURES:
            if any(n in sig_str for n in needles):
                scores[key] += 0.85
                cues.append(cue)
                declared = True
                break
    return declared


def _score_c2pa(provenance_data: Dict[str, Any], scores: Dict[str, float], cues: List[str]) -> None:
    """A Content Credentials marker says a tool signed the file, not which one: it adds weight to the generators that sign, nothing more."""
    if provenance_data.get("c2pa_present"):
        scores["openai_sora"] += 0.45
        scores["google_veo"] += 0.35
        cues.append("C2PA Content Credentials markers present in video stream (unverified)")


def _score_temporal_characteristics(temporal_data: Dict[str, Any], scores: Dict[str, float]) -> None:
    raw_var = (temporal_data.get("temporal_consistency") or {}).get("motion_variance")
    m_var = float(raw_var) if raw_var is not None else None            # None: too few frames, no reading
    if m_var is None:
        pass
    elif m_var > 140.0:
        scores["luma_dream_machine"] += 0.25
        scores["runway_gen"] += 0.20
    elif 5.0 < m_var < 30.0:
        scores["bytedance_seedance"] += 0.20
        scores["google_veo"] += 0.20
    if temporal_data.get("diffusion_flicker", {}).get("has_diffusion_flicker"):
        scores["pika"] += 0.25
        scores["kuaishou_kling"] += 0.20


class VideoModelAttributionEngine:
    """Attributes synthetic videos to specific video generative models and foundation engines."""

    def __init__(self):
        pass

    def attribute_video(
        self,
        video_path: str | Path,
        temporal_data: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        provenance_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attributes video to likely generative source."""
        path = Path(video_path)
        if not path.is_file():
            return self._unknown_attribution("File not found")

        scores: Dict[str, float] = {k: 0.05 for k in KNOWN_VIDEO_GENERATORS}
        cues: List[str] = []

        declared = _score_vendor_signatures(provenance_data, scores, cues) if provenance_data else False
        declared_keys = {k for k, v in scores.items() if v > 0.05 + 1e-9}          # what the file itself claims, before any other cue is added
        if provenance_data:
            _score_c2pa(provenance_data, scores, cues)
        if temporal_data:
            _score_temporal_characteristics(temporal_data, scores)

        return finalize_attribution(
            scores, KNOWN_VIDEO_GENERATORS, declared=declared, declared_keys=declared_keys, cues=cues,
            unknown_name="Unknown / Generic Video Diffusion", no_cue_text="No generator is declared in the file's metadata.",
            region_of=_region_for,
        )

    def attribute_media(
        self,
        media_path: str | Path,
        modality: str = "video",
        forensic_data: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        provenance_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Unified attribution interface for multi-modal workflows."""
        return self.attribute_video(
            video_path=media_path,
            temporal_data=forensic_data,
            profile_data=profile_data,
            provenance_data=provenance_data,
        )

    def _unknown_attribution(self, reason: str) -> Dict[str, Any]:
        return unknown_attribution(reason)

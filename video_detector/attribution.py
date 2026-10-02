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

        # 1. Provenance / Vendor Metadata Signatures
        # NOTE: container_atoms only ever holds generic MP4 box type names (ftyp, moov, ...)
        # and can never contain a vendor identifier -- the actual vendor/generator name
        # strings (when present) are scanned separately into vendor_signatures_found.
        if provenance_data:
            vendor_sigs = provenance_data.get("vendor_signatures_found", [])
            for sig in vendor_sigs:
                sig_str = str(sig).lower()
                if "bytedance" in sig_str or "jimeng" in sig_str:
                    scores["bytedance_seedance"] += 0.85
                    cues.append("ByteDance video metadata signature detected")
                elif "kling" in sig_str or "kuaishou" in sig_str:
                    scores["kuaishou_kling"] += 0.85
                    cues.append("Kling metadata signature detected")
                elif "runway" in sig_str or "gen-2" in sig_str or "gen-3" in sig_str:
                    scores["runway_gen"] += 0.85
                    cues.append("Runway metadata identifier detected")
                elif "sora" in sig_str or "openai" in sig_str:
                    scores["openai_sora"] += 0.85
                    cues.append("OpenAI Sora metadata signature detected")
                elif "veo" in sig_str or "deepmind" in sig_str:
                    scores["google_veo"] += 0.85
                    cues.append("Google Veo metadata signature detected")
                elif "luma" in sig_str or "dream machine" in sig_str:
                    scores["luma_dream_machine"] += 0.85
                    cues.append("Luma Dream Machine metadata signature detected")
                elif "pika" in sig_str:
                    scores["pika"] += 0.85
                    cues.append("Pika Labs metadata signature detected")
                elif "hailuo" in sig_str or "minimax" in sig_str:
                    scores["minimax_hailuo"] += 0.85
                    cues.append("MiniMax Hailuo metadata signature detected")
                elif "tongyi" in sig_str or "wanxiang" in sig_str:
                    scores["alibaba_wan"] += 0.85
                    cues.append("Alibaba Wan/Tongyi Wanxiang metadata signature detected")
                elif "hunyuan" in sig_str:
                    scores["tencent_hunyuan"] += 0.85
                    cues.append("Tencent Hunyuan Video metadata signature detected")
                elif "vidu" in sig_str or "shengshu" in sig_str:
                    scores["vidu"] += 0.85
                    cues.append("Vidu metadata signature detected")
                elif "pixverse" in sig_str:
                    scores["pixverse"] += 0.85
                    cues.append("Pixverse metadata signature detected")
                elif "movie gen" in sig_str or "moviegen" in sig_str:
                    scores["meta_movie_gen"] += 0.85
                    cues.append("Meta Movie Gen metadata signature detected")
                elif "firefly" in sig_str:
                    scores["adobe_firefly_video"] += 0.85
                    cues.append("Adobe Firefly Video metadata signature detected")
                elif "nova reel" in sig_str or "novareel" in sig_str:
                    scores["amazon_nova_reel"] += 0.85
                    cues.append("Amazon Nova Reel metadata signature detected")

            if provenance_data.get("c2pa_present"):
                scores["openai_sora"] += 0.45
                scores["google_veo"] += 0.35
                cues.append("C2PA Content Credentials signature detected in video stream")

        # 2. Temporal & Motion Characteristics
        if temporal_data:
            tc = temporal_data.get("temporal_consistency", {})
            m_var = float(tc.get("motion_variance", 0.0))
            if m_var > 140.0:
                scores["luma_dream_machine"] += 0.25
                scores["runway_gen"] += 0.20
            elif m_var < 30.0 and m_var > 5.0:
                scores["bytedance_seedance"] += 0.20
                scores["google_veo"] += 0.20

            flicker = temporal_data.get("diffusion_flicker", {})
            if flicker.get("has_diffusion_flicker"):
                scores["pika"] += 0.25
                scores["kuaishou_kling"] += 0.20

        # Normalize scores
        total = sum(scores.values())
        norm = {k: v / total for k, v in scores.items()} if total > 0 else scores
        top_candidates = sorted(norm.items(), key=lambda x: x[1], reverse=True)

        best_key, best_score = top_candidates[0]
        if best_score < 0.25:
            return {
                "attributed_model": "Unknown / Generic Video Diffusion",
                "model_key": "unknown",
                "confidence": round(best_score, 2),
                "cues": cues or ["No distinctive video foundation model fingerprints isolated."],
                "top_candidates": [{"model": KNOWN_VIDEO_GENERATORS[k]["name"], "confidence": round(s, 2)} for k, s in top_candidates[:3]],
            }

        best_info = KNOWN_VIDEO_GENERATORS[best_key]
        conf = round(best_score, 2)
        region = (
            "China" if best_key in (
                "bytedance_seedance", "kuaishou_kling", "minimax_hailuo",
                "alibaba_wan", "tencent_hunyuan", "vidu",
            )
            else "United States" if best_key in (
                "openai_sora", "google_veo", "runway_gen", "luma_dream_machine",
                "luma_dream", "pika", "meta_movie_gen", "adobe_firefly_video",
                "amazon_nova_reel",
            )
            else "Global"
        )
        return {
            "attributed_model": best_info["name"],
            "model_key": best_key,
            "provider": best_info["provider"],
            "confidence": conf,
            "attribution_confidence": conf,
            "region_of_origin": region,
            "watermark_detected": False,
            "cues": cues or [f"Temporal motion and container dynamics match {best_info['name']}"],
            "top_candidates": [{"model": KNOWN_VIDEO_GENERATORS[k]["name"], "confidence": round(s, 2)} for k, s in top_candidates[:3]],
        }

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
        return {
            "attributed_model": "Unknown",
            "model_key": "unknown",
            "confidence": 0.0,
            "attribution_confidence": 0.0,
            "region_of_origin": "Unknown",
            "watermark_detected": False,
            "cues": [reason],
            "top_candidates": [],
        }

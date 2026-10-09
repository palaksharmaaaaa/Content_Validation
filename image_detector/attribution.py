"""
image_detector.attribution: Generative AI Image Model Attribution & Fingerprinting Engine.
Profiles image characteristics against major commercial & open-source image synthesis models:
- Midjourney (v5, v6)
- OpenAI DALL-E 3 / GPT Image 1
- Black Forest Labs Flux.1 (Schnell, Dev, Pro)
- Google Gemini / Imagen 3, Gemini 2.5 Flash Image ("Nano Banana")
- Stability AI Stable Diffusion (SD 1.5, SDXL, SD 3)
- Adobe Firefly
- Leonardo.Ai
- xAI Grok Imagine / Aurora
- ByteDance Seedream
- Tencent Hunyuan Image
- Alibaba Qwen-Image
- Kuaishou Kolors
- Remini (enhancer)
Completely self-contained with zero outside dependencies.

NOTE on scoring confidence per entry: the original entries (through Canva) have
dedicated resolution-table and spectral-decay signals in addition to metadata matching. The
2025/2026 additions (Leonardo.Ai, Grok Imagine, Seedream, Hunyuan Image, Qwen-Image, Kolors,
Remini, Nano Banana) are currently metadata-signature-only -- no resolution table or spectral
calibration yet. Add real calibration for them only once backed by actual sample analysis.
"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image
from core.shared_results import unknown_attribution

logger = logging.getLogger("image_detector.attribution")

KNOWN_IMAGE_GENERATORS = {
    "google_imagen": {
        "name": "Google Gemini / Imagen 3",
        "provider": "Google DeepMind",
        "telltales": [
            "Google Gemini 4-pointed sparkle watermark",
            "IPTC trainedAlgorithmicMedia metadata",
            "Photoshop Credit: Made with Google AI",
            "SynthID invisible watermark",
            "photorealistic dynamic range",
        ],
        "resolutions": [
            (896, 1152), (896, 1200), (896, 1184), (864, 1232), (1792, 2368),
            (1024, 1024), (928, 1120), (717, 947), (1280, 896), (896, 1280),
        ],
    },
    "midjourney": {
        "name": "Midjourney (v5 / v6)",
        "provider": "Midjourney Inc.",
        "telltales": ["hyper-detailed skin microtexture", "characteristic specular highlights", "distinctive bokeh"],
        "resolutions": [(1024, 1024), (1456, 816), (816, 1456), (1792, 1024), (1024, 1792)],
    },
    "openai_dalle3": {
        "name": "OpenAI DALL-E 3",
        "provider": "OpenAI",
        "telltales": ["C2PA Content Credentials signature", "vibrant saturated illustrative palette", "smooth skin textures"],
        "resolutions": [(1024, 1024), (1792, 1024), (1024, 1792)],
    },
    "flux1": {
        "name": "Black Forest Labs Flux.1 (Schnell / Dev / Pro)",
        "provider": "Black Forest Labs",
        "telltales": ["Flow-matching Rectified Flow latent pattern", "natural typography rendering", "photorealistic hands"],
        "resolutions": [(1024, 1024), (1344, 768), (768, 1344), (1216, 832), (832, 1216)],
    },
    "stable_diffusion": {
        "name": "Stability AI Stable Diffusion (SDXL / SD 3)",
        "provider": "Stability AI",
        "telltales": ["Classifier-free guidance contrast", "latent upscaler artifact patterns", "UNet/MMDiT spectral peaks"],
        "resolutions": [(512, 512), (768, 768), (1024, 1024), (1152, 896), (896, 1152)],
    },
    "topaz_photo_ai": {
        "name": "Topaz Photo AI (Neural Restoration / Upscaler)",
        "provider": "Topaz Labs",
        "telltales": ["Topaz Photo AI metadata signature", "deep learning super-resolution", "neural edge sharpening"],
        "resolutions": [],
    },
    "ideogram2": {
        "name": "Ideogram 2.0 (Deep Learning Typography & Design)",
        "provider": "Ideogram AI",
        "telltales": ["Precise embedded graphic typography", "high dynamic range graphic layout", "coherent poster design"],
        "resolutions": [(1024, 1024), (1280, 720), (720, 1280), (1440, 960), (960, 1440)],
    },
    "recraft_v3": {
        "name": "Recraft v3 (Neural Vector & Design Engine)",
        "provider": "Recraft AI",
        "telltales": ["Clean bezier curve rasterization", "consistent vector icon palettes", "discrete brand design"],
        "resolutions": [(1024, 1024), (1820, 1024), (1024, 1820)],
    },
    "magnific_ai": {
        "name": "Magnific AI (Generative Hallucinatory Upscaler)",
        "provider": "Magnific Labs",
        "telltales": ["Hallucinatory microtexture injection", "hyper-resolution synthetic pores", "unnatural edge sharpness"],
        "resolutions": [(2048, 2048), (4096, 4096)],
    },
    "adobe_firefly": {
        "name": "Adobe Firefly (Image 3 Model)",
        "provider": "Adobe Systems",
        "telltales": ["C2PA provenance manifest", "Adobe Stock generative dataset alignment", "commercial photo safety tone"],
        "resolutions": [(2048, 2048), (1792, 1024), (1024, 1792)],
    },
    "canva": {
        "name": "Canva Graphic Design Suite",
        "provider": "Canva Pty Ltd",
        "telltales": ["Canva CreatorTool tag", "digital layout composition", "clipped transparent background"],
        "resolutions": [],
    },
    "leonardo_ai": {
        "name": "Leonardo.Ai",
        "provider": "Leonardo AI",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
    "grok_imagine": {
        "name": "xAI Grok Imagine / Aurora",
        "provider": "xAI",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
    "google_nano_banana": {
        "name": "Google Gemini 2.5 Flash Image (\"Nano Banana\")",
        "provider": "Google DeepMind",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
    "bytedance_seedream": {
        "name": "ByteDance Seedream",
        "provider": "ByteDance",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
    "tencent_hunyuan_image": {
        "name": "Tencent Hunyuan Image",
        "provider": "Tencent",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
    "alibaba_qwen_image": {
        "name": "Alibaba Qwen-Image",
        "provider": "Alibaba",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
    "kuaishou_kolors": {
        "name": "Kuaishou Kolors",
        "provider": "Kuaishou Technology",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
    "openai_gpt_image": {
        "name": "OpenAI GPT Image 1",
        "provider": "OpenAI",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
    "remini": {
        "name": "Remini (Neural Photo Enhancer)",
        "provider": "Bending Spoons",
        "telltales": ["Metadata-signature match only (no resolution/spectral fingerprint calibrated yet)"],
        "resolutions": [],
    },
}


_AUTHENTIC_PROVIDERS = {
    "AUTHENTIC_REAL_PHOTOGRAPH": "Physical Optical Camera",
    "AUTHENTIC_RECAPTURED_SCREEN": "Recaptured Physical Screen (Optical Camera)",
    "AUTHENTIC_SCREENSHOT": "Authentic Device Screen",
    "AUTHENTIC_EDITED": "Conventional Graphic Editor (Non-Generative)",
}

# (software needles, generator key, score weight, cue text). First match wins; "{sw}" is the lower-cased software string.
_SOFTWARE_SIGNATURES = (
    (("midjourney",), "midjourney", 1.2, "EXIF Software explicitly declares Midjourney ({sw})"),
    (("dall-e", "dalle"), "openai_dalle3", 1.2, "Metadata declares DALL-E generation"),
    (("stable diffusion", "automatic1111", "comfyui"), "stable_diffusion", 1.2, "Metadata declares Stable Diffusion pipeline ({sw})"),
    (("ideogram",), "ideogram2", 1.5, "Metadata declares Ideogram engine ({sw})"),
    (("recraft",), "recraft_v3", 1.5, "Metadata declares Recraft design engine ({sw})"),
    (("magnific",), "magnific_ai", 1.5, "Metadata declares Magnific AI upscaler ({sw})"),
    (("firefly",), "adobe_firefly", 1.5, "Metadata declares Adobe Firefly ({sw})"),
    (("leonardo",), "leonardo_ai", 1.5, "Metadata declares Leonardo.Ai ({sw})"),
    (("grok", "xai"), "grok_imagine", 1.5, "Metadata declares xAI Grok Imagine ({sw})"),
    (("nano banana", "gemini 2.5 flash image"), "google_nano_banana", 1.5, "Metadata declares Google Gemini 2.5 Flash Image / Nano Banana ({sw})"),
    (("seedream",), "bytedance_seedream", 1.5, "Metadata declares ByteDance Seedream ({sw})"),
    (("hunyuan",), "tencent_hunyuan_image", 1.5, "Metadata declares Tencent Hunyuan Image ({sw})"),
    (("qwen",), "alibaba_qwen_image", 1.5, "Metadata declares Alibaba Qwen-Image ({sw})"),
    (("kolors",), "kuaishou_kolors", 1.5, "Metadata declares Kuaishou Kolors ({sw})"),
    (("gpt-image", "gpt image"), "openai_gpt_image", 1.5, "Metadata declares OpenAI GPT Image 1 ({sw})"),
    (("remini",), "remini", 1.5, "Metadata declares Remini enhancement ({sw})"),
)

_REGION_BY_KEY = {
    **{k: "United States" for k in ("openai_dalle3", "midjourney", "google_imagen", "topaz_photo_ai",
                                    "leonardo_ai", "grok_imagine", "google_nano_banana", "openai_gpt_image")},
    "canva": "Australia",
    "flux1": "Germany / EU",
    "stable_diffusion": "United Kingdom",
    "remini": "Italy / EU",
    **{k: "China" for k in ("bytedance_seedream", "tencent_hunyuan_image", "alibaba_qwen_image", "kuaishou_kolors")},
}


def _authentic_attribution(tax_state: str) -> Dict[str, Any]:
    return {
        "attributed_model": "None (Authentic Capture)",
        "model_key": "none_authentic",
        "provider": _AUTHENTIC_PROVIDERS[tax_state],
        "confidence": 0.0,
        "attribution_confidence": 0.0,
        "region_of_origin": "N/A",
        "watermark_detected": False,
        "cues": ["Authentic media capture - no generative foundation model detected."],
        "top_candidates": [],
    }


def _score_declarations(
    path: Path, forensic_data: Optional[Dict[str, Any]], meta: Dict[str, Any], scores: Dict[str, float], cues: List[str]
) -> bool:
    """Direct watermark / embedded-label / software declarations (a file name is never used: anyone can name a file anything). Returns whether a watermark was seen.

    All of these are unauthenticated claims (filenames and metadata are trivially editable): they steer the
    attribution guess, which is explanation only and is never scored as evidence of synthesis.
    """
    software = str(meta.get("software", "")).lower()
    creator = str(meta.get("creator_tool", "")).lower()
    watermark = False

    if forensic_data and forensic_data.get("watermark_detected"):
        watermark = True
        scores["google_imagen"] += 1.2
        cues.append(f"Visual watermark detected: {forensic_data.get('watermark_details') or 'AI Watermark'}")
    if meta.get("photoshop_credit") == "Made with Google AI" or meta.get("iptc_digital_source_type") == "trainedAlgorithmicMedia":
        scores["google_imagen"] += 1.5
        cues.append("Embedded metadata declares: 'Made with Google AI' (trainedAlgorithmicMedia; unauthenticated label)")
    if "topaz photo ai" in software or "topaz" in creator:
        scores["topaz_photo_ai"] += 1.5
        cues.append(f"Metadata confirms enhancement software: Topaz Photo AI ({meta.get('software') or meta.get('creator_tool')})")
    if "canva" in software or "canva" in creator:
        scores["canva"] += 1.5
        cues.append(f"Metadata confirms Canva graphic design export: {meta.get('creator_tool') or 'Canva'}")
    return watermark


def _score_software_header(provenance_data: Dict[str, Any], software: str, scores: Dict[str, float], cues: List[str]) -> None:
    for needles, key, weight, cue in _SOFTWARE_SIGNATURES:
        if any(n in software for n in needles):
            scores[key] += weight
            cues.append(cue.format(sw=software))
            break
    if provenance_data.get("c2pa_present"):
        scores["openai_dalle3"] += 0.35
        scores["google_imagen"] += 0.30
        scores["adobe_firefly"] += 0.40
        cues.append("C2PA Content Credentials markers present (unverified)")


def _score_canonical_resolution(path: Path, scores: Dict[str, float], cues: List[str]) -> None:
    try:
        with Image.open(path) as img:
            w, h = img.size
        for gen_key, gen_info in KNOWN_IMAGE_GENERATORS.items():
            for rw, rh in gen_info["resolutions"]:
                if (w == rw and h == rh) or (w == rh and h == rw):
                    scores[gen_key] += 0.25
                    cues.append(f"Exact match with canonical native output resolution ({w}x{h}) of {gen_info['name']}")
                    break
    except Exception as exc:
        logger.debug("Canonical-resolution check skipped for %s: %s", path, exc)


def _score_spectral(forensic_data: Dict[str, Any], scores: Dict[str, float], cues: List[str]) -> None:
    # The measurements live in the detector's ``forensic_metrics``; a value that is not there contributes nothing.
    metrics = forensic_data.get("forensic_metrics") or {}
    decay = metrics.get("spectral_decay_alpha")
    if decay is not None:
        if float(decay) < 1.65:
            scores["midjourney"] += 0.25
            scores["flux1"] += 0.20
        elif float(decay) > 2.30:
            scores["stable_diffusion"] += 0.20
    smoothness = metrics.get("surface_smoothness")
    if smoothness is not None and float(smoothness) < 2.0:
        scores["openai_dalle3"] += 0.15
    if forensic_data.get("digital_art_detected") or metrics.get("is_digital_art"):
        scores["google_imagen"] += 0.40
        scores["openai_dalle3"] += 0.20
        scores["midjourney"] += 0.15
        cues.append("Stylistic palette and saturation profile match generative digital artwork")


class ImageModelAttributionEngine:
    """Attributes synthetic images to specific generative architectures and foundation models."""

    def __init__(self):
        pass

    def attribute_image(
        self,
        image_path: str | Path,
        forensic_data: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        provenance_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Attributes image to likely generative source."""
        path = Path(image_path)
        if not path.is_file():
            return self._unknown_attribution("File not found")

        tax_state = forensic_data.get("taxonomy_state") if forensic_data else None
        if tax_state in _AUTHENTIC_PROVIDERS:
            return _authentic_attribution(tax_state)

        scores: Dict[str, float] = {k: 0.05 for k in KNOWN_IMAGE_GENERATORS}
        cues: List[str] = []
        meta = provenance_data.get("metadata", {}) if provenance_data else {}

        watermark_detected = _score_declarations(path, forensic_data, meta, scores, cues)
        if provenance_data:
            _score_software_header(provenance_data, str(meta.get("software", "")).lower(), scores, cues)
        _score_canonical_resolution(path, scores, cues)
        if forensic_data:
            _score_spectral(forensic_data, scores, cues)

        total = sum(scores.values())
        norm_scores = {k: v / total for k, v in scores.items()} if total > 0 else scores
        top_candidates = sorted(norm_scores.items(), key=lambda x: x[1], reverse=True)
        top3 = [{"model": KNOWN_IMAGE_GENERATORS[k]["name"], "confidence": round(s, 2)} for k, s in top_candidates[:3]]

        best_key, best_score = top_candidates[0]
        if best_score < 0.25:
            return {
                "attributed_model": "Unknown / Generic Diffusion",
                "model_key": "unknown",
                "confidence": round(best_score, 2),
                "cues": cues or ["No distinctive generator-specific signatures isolated."],
                "top_candidates": top3,
            }

        best_info = KNOWN_IMAGE_GENERATORS[best_key]
        conf = round(best_score, 2)
        return {
            "attributed_model": best_info["name"],
            "model_key": best_key,
            "provider": best_info["provider"],
            "confidence": conf,
            "attribution_confidence": conf,
            "region_of_origin": _REGION_BY_KEY.get(best_key, "Global / Open-Source"),
            "watermark_detected": watermark_detected,
            "cues": cues or [f"Aesthetic, metadata, and spectral fingerprint matches {best_info['name']}"],
            "top_candidates": top3,
        }

    def _unknown_attribution(self, reason: str) -> Dict[str, Any]:
        return unknown_attribution(reason)

    def attribute_media(
        self,
        media_path: str | Path,
        modality: str = "image",
        forensic_data: Optional[Dict[str, Any]] = None,
        profile_data: Optional[Dict[str, Any]] = None,
        provenance_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Unified attribution interface for multi-modal workflows."""
        return self.attribute_image(
            image_path=media_path,
            forensic_data=forensic_data,
            profile_data=profile_data,
            provenance_data=provenance_data,
        )

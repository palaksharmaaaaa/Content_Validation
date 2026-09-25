"""
Forensic Media Memory Bank & Knowledge Registry.
Remembers media fingerprints, comprehensive pixel-level/audio specifications,
EXIF provenance, and user multi-criterion ratings (1-10 scale).
Provides fast similarity matching and learned Bayesian priors for downstream inference.
"""
from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple
import uuid

import numpy as np

from utils.logging_utils import get_logger
from utils.media_profiler import profile_media

logger = get_logger("forensic_memory")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_DIR = PROJECT_ROOT / "data" / "forensic_memory"
SAMPLES_DIR = MEMORY_DIR / "samples"
MEMORY_FILE = MEMORY_DIR / "feedback_memory.json"
CALIBRATION_FILE = MEMORY_DIR / "dynamic_calibration.json"


def ensure_storage_ready() -> None:
    """Ensures storage directories and memory stores exist."""
    MEMORY_DIR.mkdir(parents=True, exist_ok=True)
    for mod in ["image", "video", "audio"]:
        (SAMPLES_DIR / mod).mkdir(parents=True, exist_ok=True)

    if not MEMORY_FILE.is_file():
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)

    if not CALIBRATION_FILE.is_file():
        default_calibration = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "feedback_count": 0,
            "feature_weights": {
                "noise_residual": 0.35,
                "surface_smoothness": 0.30,
                "fft_decay": 0.20,
                "facial_shading": 0.25,
                "vocoder_cutoff": 0.40,
                "spectral_flatness": 0.30,
                "silence_ratio": 0.25,
            },
            "sensitivity_offsets": {
                "noise_center_offset": 0.0,
                "smooth_center_offset": 0.0,
                "audio_ai_offset": 0.0,
            },
            "learned_clusters": [],
        }
        with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
            json.dump(default_calibration, f, indent=2)


def extract_feature_vector(modality: str, forensic_data: Dict[str, Any], profile_data: Dict[str, Any]) -> List[float]:
    """
    Builds a normalized, comparable 8-dimensional forensic embedding vector
    for similarity lookup across media files.
    """
    mod = modality.lower()
    if mod in ("image", "video"):
        f_metrics = forensic_data.get("forensic_metrics", {})
        noise = float(f_metrics.get("noise_residual_mean", 2.2))
        smooth = float(f_metrics.get("surface_smoothness", 3.2))
        alpha = float(f_metrics.get("spectral_decay_alpha", 2.0))
        ela = float(f_metrics.get("ela_mean_error", 2.0))

        p_spec = profile_data.get("pixel_specifications", {})
        if not p_spec and profile_data.get("stream_specifications"):
            p_spec = profile_data["stream_specifications"]

        entropy = float(p_spec.get("shannon_entropy_bpp", p_spec.get("keyframe_entropy_bpp", 7.0)))
        bpp = float(p_spec.get("bits_per_pixel", 24))
        aspect = float(p_spec.get("aspect_ratio_decimal", 1.0))
        var = float(p_spec.get("pixel_variance", 1000.0)) / 10000.0

        # Normalized feature components
        return [
            np.clip(noise / 5.0, 0.0, 1.0),
            np.clip(smooth / 8.0, 0.0, 1.0),
            np.clip(alpha / 4.0, 0.0, 1.0),
            np.clip(ela / 10.0, 0.0, 1.0),
            np.clip(entropy / 8.0, 0.0, 1.0),
            np.clip(bpp / 48.0, 0.0, 1.0),
            np.clip(aspect / 3.0, 0.0, 1.0),
            np.clip(var, 0.0, 1.0),
        ]
    else:  # audio
        s_feats = forensic_data.get("spectral_features", {})
        sig_spec = profile_data.get("signal_specifications", {})

        has_cut = 1.0 if s_feats.get("has_vocoder_cutoff") else 0.0
        cut_hz = float(s_feats.get("cutoff_freq_hz", 8000)) / 16000.0
        flatness = float(s_feats.get("spectral_flatness", 0.02)) * 20.0
        silence = float(s_feats.get("digital_silence_ratio", 0.0))
        rms = float(sig_spec.get("rms_energy", 0.1)) * 5.0
        crest = float(sig_spec.get("crest_factor", 3.0)) / 10.0
        zcr = float(sig_spec.get("zero_crossing_rate", 0.05)) * 10.0
        entropy = float(sig_spec.get("audio_entropy", 5.0)) / 8.0

        return [
            np.clip(has_cut, 0.0, 1.0),
            np.clip(cut_hz, 0.0, 1.0),
            np.clip(flatness, 0.0, 1.0),
            np.clip(silence, 0.0, 1.0),
            np.clip(rms, 0.0, 1.0),
            np.clip(crest, 0.0, 1.0),
            np.clip(zcr, 0.0, 1.0),
            np.clip(entropy, 0.0, 1.0),
        ]


class ForensicMemory:
    """Persistent Forensic Knowledge Store and Retrieval Engine."""

    def __init__(self):
        ensure_storage_ready()

    def load_memory(self) -> List[Dict[str, Any]]:
        ensure_storage_ready()
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load memory file: %s", exc)
            return []

    def save_memory(self, memory_records: List[Dict[str, Any]]) -> bool:
        ensure_storage_ready()
        try:
            with open(MEMORY_FILE, "w", encoding="utf-8") as f:
                json.dump(memory_records, f, indent=2)
            return True
        except Exception as exc:
            logger.error("Failed to save memory file: %s", exc)
            return False

    def load_calibration(self) -> Dict[str, Any]:
        ensure_storage_ready()
        try:
            with open(CALIBRATION_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Failed to load calibration: %s", exc)
            return {}

    def save_calibration(self, calibration_data: Dict[str, Any]) -> bool:
        ensure_storage_ready()
        try:
            with open(CALIBRATION_FILE, "w", encoding="utf-8") as f:
                json.dump(calibration_data, f, indent=2)
            return True
        except Exception as exc:
            logger.error("Failed to save calibration: %s", exc)
            return False

    def register_feedback(
        self,
        media_path: str | Path,
        modality: str,
        ratings: Dict[str, float],
        ground_truth: str,
        forensic_data: Dict[str, Any],
        user_notes: str = "",
        generator_tag: str = "",
    ) -> Dict[str, Any]:
        """
        Stores user ratings (out of 10) along with complete media specifications,
        hashes, pixel attributes, and forensic vectors into the memory bank.
        Triggers end-to-end auto-updating.
        """
        from learning.auto_learner import AutoLearner

        media_path = Path(media_path)
        ensure_storage_ready()

        # 1. Profile media
        profile_data = profile_media(media_path, modality=modality)
        file_id = profile_data.get("file_identity", {})
        sha256 = file_id.get("sha256", uuid.uuid4().hex)
        ext = file_id.get("extension", media_path.suffix.lower() or ".dat")

        # 2. Archive media file into samples pool for continual learning
        archived_name = f"{sha256[:16]}_{int(datetime.now().timestamp())}{ext}"
        archived_dest = SAMPLES_DIR / modality.lower() / archived_name
        try:
            if not archived_dest.is_file():
                shutil.copy2(str(media_path), str(archived_dest))
        except Exception as exc:
            logger.warning("Could not archive media sample: %s", exc)

        # 3. Compute comparable feature vector
        vector = extract_feature_vector(modality, forensic_data, profile_data)

        # 4. Construct record
        record_id = str(uuid.uuid4())
        record = {
            "id": record_id,
            "timestamp": datetime.now().isoformat(),
            "modality": modality.lower(),
            "ratings": {
                "real_score": round(float(ratings.get("real_score", 5.0)), 1),
                "synthetic_score": round(float(ratings.get("synthetic_score", 5.0)), 1),
                "forged_score": round(float(ratings.get("forged_score", 5.0)), 1),
                "undetected_score": round(float(ratings.get("undetected_score", 0.0)), 1),
            },
            "ground_truth": ground_truth,
            "generator_tag": generator_tag.strip(),
            "user_notes": user_notes.strip(),
            "file_identity": file_id,
            "specifications": profile_data.get("pixel_specifications") or profile_data.get("signal_specifications") or profile_data.get("stream_specifications") or {},
            "provenance_metadata": profile_data.get("provenance_metadata", {}),
            "forensic_metrics": forensic_data.get("forensic_metrics") or forensic_data.get("spectral_features") or {},
            "archived_sample_path": str(archived_dest.relative_to(PROJECT_ROOT)),
            "feature_vector": vector,
        }

        memory = self.load_memory()
        memory.append(record)
        self.save_memory(memory)
        logger.info("Registered forensic feedback for %s [%s] -> ID: %s", media_path.name, modality, record_id)

        # 5. Dynamically auto-update model parameters and weights end-to-end
        learner = AutoLearner()
        update_res = learner.on_feedback_received(record, memory)

        return {
            "success": True,
            "record_id": record_id,
            "sha256": sha256,
            "archived_path": str(archived_dest),
            "auto_update": update_res,
        }

    def query_similar_media(
        self,
        media_path: str | Path,
        modality: str,
        forensic_data: Dict[str, Any],
        top_k: int = 4,
        similarity_threshold: float = 0.72,
    ) -> Dict[str, Any]:
        """
        Inspects the Forensic Memory Bank for exact cryptographic matches
        or high-confidence feature vector similarity.
        Returns matched records and computes a dynamic Bayesian prior adjustment.
        """
        memory = self.load_memory()
        if not memory:
            return {"has_matches": False, "matches": [], "prior_adjustment": {"ai_boost": 0.0, "real_boost": 0.0}}

        profile_data = profile_media(media_path, modality=modality)
        target_sha = profile_data.get("file_identity", {}).get("sha256")
        target_vec = np.array(extract_feature_vector(modality, forensic_data, profile_data), dtype=np.float32)

        norm_target = np.linalg.norm(target_vec)
        if norm_target > 1e-6:
            target_vec = target_vec / norm_target

        matches = []
        for rec in memory:
            if rec.get("modality") != modality.lower():
                continue

            rec_sha = rec.get("file_identity", {}).get("sha256")
            exact_hash = (target_sha is not None and rec_sha == target_sha)

            rec_vec = np.array(rec.get("feature_vector", []), dtype=np.float32)
            if len(rec_vec) != len(target_vec):
                sim = 1.0 if exact_hash else 0.5
            else:
                norm_rec = np.linalg.norm(rec_vec)
                if norm_rec > 1e-6:
                    rec_vec = rec_vec / norm_rec
                sim = float(np.dot(target_vec, rec_vec))

            if exact_hash:
                sim = 1.0

            if sim >= similarity_threshold or exact_hash:
                matches.append({
                    "id": rec.get("id"),
                    "similarity": round(float(sim), 3),
                    "exact_hash_match": exact_hash,
                    "ground_truth": rec.get("ground_truth"),
                    "ratings": rec.get("ratings", {}),
                    "generator_tag": rec.get("generator_tag"),
                    "user_notes": rec.get("user_notes"),
                    "specifications": rec.get("specifications", {}),
                    "timestamp": rec.get("timestamp"),
                })

        matches.sort(key=lambda m: m["similarity"], reverse=True)
        top_matches = matches[:top_k]

        if not top_matches:
            return {"has_matches": False, "matches": [], "prior_adjustment": {"ai_boost": 0.0, "real_boost": 0.0}}

        # Aggregate learned prior from similar cases
        sim_weights = [m["similarity"] for m in top_matches]
        total_weight = sum(sim_weights) or 1.0

        avg_synth_rating = sum(m["ratings"].get("synthetic_score", 5.0) * m["similarity"] for m in top_matches) / total_weight
        avg_real_rating = sum(m["ratings"].get("real_score", 5.0) * m["similarity"] for m in top_matches) / total_weight

        # Scale rating difference (out of 10) into Bayesian evidence boost [-0.40, +0.40]
        rating_delta = (avg_synth_rating - avg_real_rating) / 10.0  # -1.0 to 1.0
        ai_boost = float(np.clip(rating_delta * 0.35 * max(sim_weights), -0.40, 0.40))
        real_boost = float(-ai_boost)

        explanation = (
            f"Forensic Memory matched {len(top_matches)} similar profile(s) "
            f"(Top similarity: {top_matches[0]['similarity']*100:.1f}%). "
            f"Learned historical rating: {avg_synth_rating:.1f}/10 Synthetic vs {avg_real_rating:.1f}/10 Real."
        )

        return {
            "has_matches": True,
            "match_count": len(top_matches),
            "top_similarity": top_matches[0]["similarity"],
            "exact_hash_found": any(m["exact_hash_match"] for m in top_matches),
            "avg_synthetic_rating": round(avg_synth_rating, 2),
            "avg_real_rating": round(avg_real_rating, 2),
            "prior_adjustment": {
                "ai_boost": round(ai_boost, 3),
                "real_boost": round(real_boost, 3),
            },
            "explanation": explanation,
            "matches": top_matches,
        }

    def get_memory_stats(self) -> Dict[str, Any]:
        """Provides summary metrics for the continuous learning dashboard."""
        memory = self.load_memory()
        calibration = self.load_calibration()

        total_samples = len(memory)
        by_modality = {"image": 0, "video": 0, "audio": 0}
        ratings_sum = {"real": 0.0, "synthetic": 0.0, "forged": 0.0, "undetected": 0.0}

        for rec in memory:
            mod = rec.get("modality", "image")
            by_modality[mod] = by_modality.get(mod, 0) + 1
            r = rec.get("ratings", {})
            ratings_sum["real"] += r.get("real_score", 0.0)
            ratings_sum["synthetic"] += r.get("synthetic_score", 0.0)
            ratings_sum["forged"] += r.get("forged_score", 0.0)
            ratings_sum["undetected"] += r.get("undetected_score", 0.0)

        n = max(1, total_samples)
        return {
            "total_samples": total_samples,
            "by_modality": by_modality,
            "average_ratings": {
                "real": round(ratings_sum["real"] / n, 2),
                "synthetic": round(ratings_sum["synthetic"] / n, 2),
                "forged": round(ratings_sum["forged"] / n, 2),
                "undetected": round(ratings_sum["undetected"] / n, 2),
            },
            "active_weights": calibration.get("feature_weights", {}),
            "sensitivity_offsets": calibration.get("sensitivity_offsets", {}),
            "last_updated": calibration.get("last_updated"),
        }

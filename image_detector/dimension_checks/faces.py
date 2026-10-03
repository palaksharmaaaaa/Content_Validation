"""
image_detector.dimension_checks.faces: per-face authenticity (real photograph or generated face).

A learned signal with a small, one-directional effect: a face that looks AI-generated adds a capped amount of evidence
toward AI (and blocks a "likely real" verdict); a face that looks real adds nothing, because a generator the classifier
has never seen would also look "real" to it. It uses the YuNet face detector and the face
classifier trained by ``image_detector.face_training``; both are reported honestly when unavailable.
"""
from __future__ import annotations

import cv2
from core.imageio import imread

from core.face_detection import get_face_finder
from core.forensics.registry import CheckContext, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity
from image_detector.face_authenticity import MIN_FACE_PIXELS, get_face_authenticity

AI_LIKE_LLR = 0.40    # base-10 log-likelihood ratio added toward AI (the whole hybrid-scoring budget)
DOMINANT_AREA = 0.15   # a face covering this fraction of the image makes the face the main subject
AI_LIKE = 0.90        # a face scored at or above this is called AI-like
UNSURE = 0.50
MAX_SIDE = 2000       # large photos are downscaled before cropping; faces stay well above MIN_FACE_PIXELS


def _finding(status: FindingStatus, severity: Severity, detail: str, data: dict, llr: float = None) -> Finding:
    return Finding(check_id="face_authenticity", dimension="Faces", stage="faces", title="Face authenticity",
                   status=status, severity=severity, evidence_class=EvidenceClass.LEARNED_SIGNAL, detail=detail, data=data, llr=llr, llr_cap=AI_LIKE_LLR)


@registry.register("image", "face_authenticity", phase="pre")
def check_face_authenticity(ctx: CheckContext) -> Finding:
    classifier = get_face_authenticity()
    if not classifier.available:
        return _finding(FindingStatus.NOT_CALIBRATED, Severity.NONE,
                        "No trained face model. Train one with: python -m image_detector.face_training --real <folder> --ai <folder>.", {})
    img = imread(str(ctx.path), cv2.IMREAD_COLOR)
    if img is None:
        return _finding(FindingStatus.NOT_APPLICABLE, Severity.NONE, "The image could not be decoded for face analysis.", {})
    scale = MAX_SIDE / max(img.shape[:2])
    if scale < 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    faces = get_face_finder().find(img)
    if not faces:
        return _finding(FindingStatus.NOT_APPLICABLE, Severity.NONE, "No face was found.", {"faces_found": 0})
    result = classifier.analyse(img, faces)
    judged = result["faces"]
    data = {"faces_found": len(faces), "judged": judged, "skipped_small": result["skipped_small"]}
    if not judged:
        return _finding(FindingStatus.INFO, Severity.NONE,
                        f"{len(faces)} face(s) found, all smaller than {MIN_FACE_PIXELS} px, too small to judge.", data)
    worst = max(f["p_ai"] for f in judged)
    summary = ", ".join(f"{f['p_ai'] * 100:.0f}%" for f in judged)
    img_area = float(img.shape[0] * img.shape[1])
    biggest = max(f[2] * f[3] for f in faces) / img_area
    data.update(largest_face_area_fraction=round(biggest, 3), worst_p_ai=worst, face_dominant=biggest >= DOMINANT_AREA)
    skipped = f" {result['skipped_small']} smaller face(s) were not judged." if result["skipped_small"] else ""
    note = " Trained on one public dataset."
    if worst >= AI_LIKE:
        return _finding(FindingStatus.WARN, Severity.MEDIUM,
                        f"{len(judged)} face(s) judged; chance of being AI-generated: {summary}. At least one looks generated.{skipped}{note}", {**data, "ai_like": True, "worst_p_ai": worst, "face_dominant": biggest >= DOMINANT_AREA}, llr=AI_LIKE_LLR)
    if worst >= UNSURE:
        return _finding(FindingStatus.INFO, Severity.LOW,
                        f"{len(judged)} face(s) judged; chance of being AI-generated: {summary}. Uncertain.{skipped}{note}", data)
    return _finding(FindingStatus.PASS, Severity.NONE,
                    f"{len(judged)} face(s) judged; chance of being AI-generated: {summary}. They look like real photographs.{skipped}{note}", data)

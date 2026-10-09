"""core.perception.age: apparent-age estimation and minor screening for people in an image.

Model: MiVOLO v2 (``iitolstykh/mivolo_v2``, Apache-2.0, 28.8 M parameters, pinned revision below). It reads a face crop and
a body crop together, so a person whose face is small, turned away or hidden can still be aged from the body. Either crop may
be missing. A second, independent opinion comes from the SigLIP 2 model the project already loads: the same crop is scored
against age-group descriptions (core.perception.vocab.AGE_GROUPS).

The screening is recall-first, because missing a child is far worse than flagging an adult. A person is therefore reported as:

  * LIKELY_MINOR    estimated age below 18
  * POSSIBLE_MINOR  estimated age below ``POSSIBLE_MINOR_AGE`` (18 plus a safety margin), or the age-group opinion puts at
                    least half its weight on baby / toddler / child / teenager
  * UNDETERMINED    a person was found but neither model had a usable crop (too small / cut off): needs a human look
  * ADULT           everything else

No model reaches 100 % recall on minors, and the margin is a trade-off: a 17-year-old and a 19-year-old look alike, so catching
nearly every minor means flagging many young adults. Measured on 2,250 UTKFace faces (services/age_eval.py): estimating
"age < 18" alone finds only 85.6 % of minors (46.7 % of 16-17 year-olds); "age < 27" finds 99.2 % (97.7 % of 16-17) and also
flags about 87 % of 18-20, 75 % of 21-25, 26 % of 26-35 and 1 % of over-35s. Those are pre-cropped, well-lit faces, so real photographs will
do worse (see docs/LIMITATIONS.md). Any image with ``review_required`` should go to a person. If the age model cannot load, the
result says ``status: UNAVAILABLE`` and ``review_required: True``; it never reports "no minors" without having checked.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from core.perception.hub import silence_noise

silence_noise()
logger = logging.getLogger("core.perception.age")

MODEL_ID = "iitolstykh/mivolo_v2"
MODEL_REVISION = "53393526c220e34cdd7b722b36d22b6f9e5f4241"
WEIGHTS_PATH = Path(__file__).resolve().parents[1] / "models" / "mivolo_v2" / "model.safetensors"
INPUT_SIZE = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MIN_AGE, MAX_AGE, AVG_AGE = 0.0, 122.0, 61.0           # from the model's config.json: age = raw * (max - min) + avg

MINOR_AGE = 18.0
POSSIBLE_MINOR_AGE = 27.0       # chosen on UTKFace: 99.2 % of under-18s fall below it (25 gave 97.8 %, 30 gave 99.8 %)
MINOR_GROUP_POSSIBLE = 0.50     # age-group opinion: share of baby..teenager above which a person is at least a possible minor
MIN_FACE_SIDE = 28              # a face crop smaller than this (px) is too coarse to age on its own
MIN_BODY_SIDE = 48
MINOR_LABELS = ("baby", "toddler", "child", "teenager")
_CROP_PAD = 0.10

Box = Tuple[int, int, int, int]  # x, y, w, h


def _letterbox_normalise(crop_bgr: Optional[np.ndarray]) -> np.ndarray:
    """MiVOLO's own preprocessing: fit into a black-padded 384 x 384 square, BGR to RGB, ImageNet normalisation. A missing
    crop becomes the all-black image after normalisation, exactly as the reference code does. Returns CHW float32."""
    if crop_bgr is None or crop_bgr.size == 0:
        arr = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), dtype=np.float32)
    else:
        h, w = crop_bgr.shape[:2]
        r = min(INPUT_SIZE / h, INPUT_SIZE / w)
        new_w, new_h = int(round(w * r)), int(round(h * r))
        if (new_w, new_h) != (w, h):
            crop_bgr = cv2.resize(crop_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        dw, dh = (INPUT_SIZE - new_w) / 2, (INPUT_SIZE - new_h) / 2
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        padded = cv2.copyMakeBorder(crop_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        arr = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.ascontiguousarray(((arr - MEAN) / STD).transpose(2, 0, 1))


def _crop(image: np.ndarray, box: Box, pad: float = _CROP_PAD) -> np.ndarray:
    h, w = image.shape[:2]
    x, y, bw, bh = box
    px, py = int(bw * pad), int(bh * pad)
    return image[max(0, y - py): min(h, y + bh + py), max(0, x - px): min(w, x + bw + px)]


def pair_faces_with_persons(faces: Sequence[Box], persons: Sequence[Box]) -> List[Tuple[Optional[int], Optional[int]]]:
    """(face index, person index) for every subject. A face joins the smallest person box that contains its centre; a person
    takes at most one face (the largest). Faces without a person and persons without a face stay as one-sided subjects."""
    taken: Dict[int, int] = {}
    unmatched_faces: List[int] = []
    for fi in sorted(range(len(faces)), key=lambda i: -(faces[i][2] * faces[i][3])):
        fx, fy, fw, fh = faces[fi]
        cx, cy = fx + fw / 2, fy + fh / 2
        owners = [pi for pi, (px, py, pw, ph) in enumerate(persons) if px <= cx <= px + pw and py <= cy <= py + ph and pi not in taken]
        if owners:
            taken[min(owners, key=lambda i: persons[i][2] * persons[i][3])] = fi
        else:
            unmatched_faces.append(fi)
    pairs: List[Tuple[Optional[int], Optional[int]]] = [(fi, pi) for pi, fi in taken.items()]
    pairs += [(fi, None) for fi in unmatched_faces]
    pairs += [(None, pi) for pi in range(len(persons)) if pi not in taken]
    return pairs


def classify_minor(age: Optional[float], minor_group_share: Optional[float], usable: bool) -> str:
    """The recall-first decision rule (see module doc). ``usable`` is False when no crop was big enough to judge."""
    if age is not None and age < MINOR_AGE:
        return "LIKELY_MINOR"
    if age is not None and age < POSSIBLE_MINOR_AGE:
        return "POSSIBLE_MINOR"
    if minor_group_share is not None and minor_group_share >= MINOR_GROUP_POSSIBLE:
        return "POSSIBLE_MINOR"
    return "ADULT" if usable else "UNDETERMINED"


class AgeEstimator:
    """Loads MiVOLO v2 on first use; thread-safe. ``assess`` never raises: failures come back as ``status: UNAVAILABLE``."""

    def __init__(self, weights: Path = WEIGHTS_PATH):
        self._weights = Path(weights)
        self._model = None
        self._lock = threading.Lock()
        self._failed = False

    def _ensure(self):
        with self._lock:
            if self._model is None and not self._failed:
                try:
                    from safetensors.torch import load_file

                    from core.perception.hub import ensure_not_lfs_pointer

                    ensure_not_lfs_pointer(self._weights)

                    from core.perception.mivolo_vendor.mivolo_net import MiVOLOModel

                    net = MiVOLOModel(layers=(4, 4, 8, 2), img_size=INPUT_SIZE, in_chans=6, num_classes=3,
                                      embed_dims=(192, 384, 384, 384), num_heads=(6, 12, 12, 12))
                    state = {k.replace("mivolo.model.", "", 1): v.float() for k, v in load_file(str(self._weights)).items()}
                    net.load_state_dict(state, strict=True)
                    self._model = net.eval().requires_grad_(False)       # frozen here only; autograd stays on for the rest of the process
                except Exception as exc:
                    logger.warning("Age model unavailable (%s): %s", self._weights, exc)
                    self._failed = True
            return self._model

    @property
    def available(self) -> bool:
        """True once the model has loaded."""
        return self._ensure() is not None

    def _ages(self, face_crops: Sequence[Optional[np.ndarray]], body_crops: Sequence[Optional[np.ndarray]]) -> List[float]:
        import torch

        batch = np.stack([np.concatenate([_letterbox_normalise(f), _letterbox_normalise(b)], axis=0) for f, b in zip(face_crops, body_crops)])
        with torch.inference_mode():
            out = self._model(torch.from_numpy(batch))
        raw = out[:, 2].float().numpy()                                  # columns 0-1 are the gender logits, which are not used
        return [float(np.clip(r * (MAX_AGE - MIN_AGE) + AVG_AGE, MIN_AGE, MAX_AGE)) for r in raw]

    @staticmethod
    def _age_group_shares(crops_bgr: Sequence[np.ndarray]) -> List[Optional[float]]:
        """Share of child + teenager in the SigLIP 2 age-group opinion for each crop (None if SigLIP is unavailable)."""
        from core.perception.recognizer import get_recognizer

        rec = get_recognizer()
        if not crops_bgr or not rec.available:
            return [None] * len(crops_bgr)
        rgb = [cv2.cvtColor(c, cv2.COLOR_BGR2RGB) for c in crops_bgr]
        tops = rec.classify(rgb, "age_group", top_k=8)
        return [round(sum(p for label, p in row if label in MINOR_LABELS), 3) if row else None for row in tops]

    @staticmethod
    def _rotated_face(image_bgr: np.ndarray, person: Box) -> Optional[Dict[str, Any]]:
        """A sideways or upside-down face inside a person's box that the upright finder missed: the box is turned until the face
        is upright, and the turned box also becomes the body crop. None if no such face is found."""
        from core.perception.face_scan import ROTATIONS, get_screening_finder

        height, width = image_bgr.shape[:2]
        x, y, w, h = person
        px, py = int(w * 0.05), int(h * 0.05)
        x0, y0, x1, y1 = max(0, x - px), max(0, y - py), min(width, x + w + px), min(height, y + h + py)
        region = image_bgr[y0:y1, x0:x1]
        if region.size == 0 or min(region.shape[:2]) < MIN_BODY_SIDE:
            return None
        hits = get_screening_finder().find_rotated(region)
        if not hits:
            return None
        best = max(hits, key=lambda hit: hit["score"])
        turned = cv2.rotate(region, ROTATIONS[best["rot"]])
        face = _crop(turned, best["rot_box"])
        if face.size == 0 or min(face.shape[:2]) < MIN_FACE_SIDE:
            return None
        bx, by, bw, bh = best["box"]
        return {"rot": best["rot"], "face": face, "body": turned, "face_box": [bx + x0, by + y0, bw, bh]}

    def assess(self, image_bgr: np.ndarray, faces: Optional[Sequence[Box]] = None, persons: Optional[Sequence[Box]] = None,
               scan_rotated: bool = True) -> Dict[str, Any]:
        """Age and minor screening for everyone in the image. ``faces`` / ``persons`` are (x, y, w, h) boxes in the pixels of
        ``image_bgr``; when omitted they are detected here (YuNet faces, RF-DETR persons). A person with no upright face is also
        searched for a sideways or upside-down one (``scan_rotated``); if one is found the person is aged both ways and the younger
        age and the larger child share are kept, so a false face can only add a review, never remove one."""
        base: Dict[str, Any] = {"model": f"{MODEL_ID}@{MODEL_REVISION[:7]}", "subjects": [], "n_subjects": 0, "youngest_age": None,
                                "contains_minor": False, "contains_possible_minor": False, "review_required": False}
        if image_bgr is None or image_bgr.ndim != 3:
            return {**base, "status": "NO_IMAGE"}
        if faces is None:
            from core.perception.face_scan import get_screening_finder

            faces = get_screening_finder().find(image_bgr)
        if persons is None:
            from core.perception.detector import get_object_detector

            persons = [d["box"] for d in get_object_detector().detect(image_bgr) if d["label"] == "person"]
        faces, persons = [tuple(int(v) for v in b) for b in faces], [tuple(int(v) for v in b) for b in persons]
        pairs = pair_faces_with_persons(faces, persons)
        if not pairs:
            return {**base, "status": "NO_PEOPLE"}
        if self._ensure() is None:
            return {**base, "status": "UNAVAILABLE", "n_subjects": len(pairs), "review_required": True,
                    "reason": "age model could not be loaded; people were found but not aged"}

        # Every subject is judged from one or more (face crop, body crop) variants; the youngest age and the largest child share win.
        variants: List[Tuple[int, Optional[np.ndarray], Optional[np.ndarray]]] = []
        evidence: List[str] = []
        face_boxes: List[Optional[List[int]]] = []
        rotation: List[int] = []
        for i, (fi, pi) in enumerate(pairs):
            fc = _crop(image_bgr, faces[fi]) if fi is not None else None
            bc = _crop(image_bgr, persons[pi], pad=0.03) if pi is not None else None
            if fc is not None and min(fc.shape[:2]) < MIN_FACE_SIDE:
                fc = None
            if bc is not None and min(bc.shape[:2]) < MIN_BODY_SIDE:
                bc = None
            evidence.append("face+body" if fc is not None and bc is not None else "face" if fc is not None else "body" if bc is not None else "none")
            face_boxes.append(None if fi is None else list(faces[fi]))
            rotation.append(0)
            if fc is not None or bc is not None:
                variants.append((i, fc, bc))
            if scan_rotated and fi is None and pi is not None:
                turned = self._rotated_face(image_bgr, persons[pi])
                if turned is not None:
                    variants.append((i, turned["face"], turned["body"]))
                    evidence[i], face_boxes[i], rotation[i] = "face+body (rotated face)", turned["face_box"], turned["rot"]

        ages: Dict[int, float] = {}
        shares: Dict[int, Optional[float]] = {}
        if variants:
            for (i, _f, _b), a in zip(variants, self._ages([v[1] for v in variants], [v[2] for v in variants])):
                ages[i] = min(a, ages.get(i, a))
            group_crops = [v[1] if v[1] is not None else v[2] for v in variants]
            for (i, _f, _b), sh in zip(variants, self._age_group_shares(group_crops)):
                if sh is not None:
                    shares[i] = max(sh, shares.get(i, sh))
                else:
                    shares.setdefault(i, None)

        subjects = []
        for i, (fi, pi) in enumerate(pairs):
            age, share = ages.get(i), shares.get(i)
            subjects.append({
                "age": None if age is None else round(age, 1),
                "minor_group_share": share,
                "assessment": classify_minor(age, share, usable=i in ages),
                "evidence": evidence[i],
                "rotation": rotation[i],
                "face_box": face_boxes[i],
                "person_box": None if pi is None else list(persons[pi]),
            })
        known = [s["age"] for s in subjects if s["age"] is not None]
        flags = {s["assessment"] for s in subjects}
        return {**base, "status": "OK", "subjects": subjects, "n_subjects": len(subjects),
                "youngest_age": min(known) if known else None,
                "contains_minor": "LIKELY_MINOR" in flags,
                "contains_possible_minor": bool(flags & {"LIKELY_MINOR", "POSSIBLE_MINOR"}),
                "review_required": bool(flags & {"LIKELY_MINOR", "POSSIBLE_MINOR", "UNDETERMINED"})}


_default: Optional[AgeEstimator] = None
_default_lock = threading.Lock()


def get_age_estimator() -> AgeEstimator:
    """The process-wide estimator."""
    global _default
    with _default_lock:
        if _default is None:
            _default = AgeEstimator()
        return _default

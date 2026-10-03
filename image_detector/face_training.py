"""
image_detector.face_training: train the face-authenticity classifier from folders of face crops.

    python -m image_detector.face_training --real "D:/.../Real Images" --ai "D:/.../AI-Generated Images" --epochs 6

Files are read in place (nothing is copied). The train/validation split comes from each file's content hash
(``core.media_library.partition``), so validation faces are never trained on and the split never reshuffles.

Shortcut control. Public face datasets usually differ between classes in ways unrelated to faces (image size, JPEG
quality, EXIF). A network will happily learn those instead. Every training image therefore gets a random
down/up-scale and a random JPEG re-encode, for both classes, and the report includes accuracy on validation images
degraded the same way for both classes ("stress" accuracy). Trust the stress number more than the clean one; the
checkpoint with the best stress accuracy is kept.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image

from core.hashing import file_sha256
from core.media_library import partition
from image_detector.face_authenticity import CHECKPOINT, INPUT_SIZE, build_model, crop_face, crop_region, to_tensor

logger = logging.getLogger("image_detector.face_training")
_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def list_files(folder: Path) -> List[Path]:
    """Image files under ``folder``, sorted."""
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in _EXT)


def split(files: List[Path]) -> Tuple[List[Path], List[Path]]:
    """(train, validation) by content hash."""
    train, val = [], []
    for p in files:
        (val if partition(file_sha256(p, cached=False)) == "val" else train).append(p)
    return train, val


def _jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def find_boxes(files: List[Path]) -> Dict[str, Tuple[int, int, int, int]]:
    """The most confident face box per file, found with the same detector inference uses. Files with no face are omitted."""
    import cv2

    from core.face_detection import get_face_finder

    finder, boxes = get_face_finder(), {}
    for p in files:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        faces = finder.find(img) if img is not None else []
        if faces:
            boxes[str(p)] = tuple(int(v) for v in faces[0][:4])
    return boxes


def prepare(path: Path, box: Tuple[int, int, int, int], train: bool, stress: bool = False) -> np.ndarray:
    """The INPUT_SIZE RGB face crop for one training/validation image, built exactly as at inference
    (detector box + margin, ``crop_region``), plus the shortcut-removing degradations when requested."""
    import cv2

    rng = random.Random() if train else random.Random(str(path))
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    x, y, w, h = box
    if train:                                             # box jitter: the detector's box is never identical twice
        s = rng.uniform(0.88, 1.15)
        dx, dy = rng.uniform(-0.08, 0.08) * w, rng.uniform(-0.08, 0.08) * h
        x, y, w, h = int(x + dx - (s - 1) * w / 2), int(y + dy - (s - 1) * h / 2), max(8, int(w * s)), max(8, int(h * s))
    crop = Image.fromarray(cv2.cvtColor(crop_region(img, (x, y, w, h)), cv2.COLOR_BGR2RGB))
    if train or stress:
        mid = rng.randint(56, 224)                        # random intermediate resolution: size is not a cue
        crop = crop.resize((mid, mid), Image.BILINEAR)
        if rng.random() < 0.8:
            crop = _jpeg(crop, rng.randint(40, 95))      # JPEG quality is not a cue
    arr = np.asarray(crop.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR), dtype=np.uint8)
    if train and rng.random() < 0.5:
        arr = arr[:, ::-1]
    return arr


class FaceSet:
    """A list of (path, label) as a torch dataset."""

    def __init__(self, items: List[Tuple[Path, int]], boxes: Dict[str, Tuple[int, int, int, int]], train: bool, stress: bool = False):
        self.items, self.boxes, self.train, self.stress = items, boxes, train, stress

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label = self.items[i]
        return to_tensor(prepare(path, self.boxes[str(path)], self.train, self.stress)), label


def evaluate(model, dataset: FaceSet, batch: int = 128, workers: int = 4) -> Dict[str, float]:
    """Accuracy, AI recall and real specificity at a 0.5 threshold."""
    import torch
    from torch.utils.data import DataLoader

    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in DataLoader(dataset, batch_size=batch, num_workers=workers):
            ps += torch.softmax(model(x), 1)[:, 1].tolist()
            ys += y.tolist()
    ys, ps = np.array(ys), np.array(ps)
    pred = ps >= 0.5
    ai, real = ys == 1, ys == 0
    return {
        "accuracy": float((pred == ai).mean()),
        "ai_recall": float(pred[ai].mean()) if ai.any() else float("nan"),
        "real_specificity": float((~pred[real]).mean()) if real.any() else float("nan"),
        "n": int(len(ys)),
    }


def evaluate_inference(checkpoint: Path, items: List[Tuple[Path, int]]) -> Dict[str, float]:
    """Accuracy through the exact inference path (``FaceAuthenticityClassifier.analyse``) on untouched validation images."""
    import cv2

    from core.face_detection import get_face_finder
    from image_detector.face_authenticity import FaceAuthenticityClassifier

    clf, finder = FaceAuthenticityClassifier(checkpoint), get_face_finder()
    ys, ps = [], []
    for path, label in items:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        res = clf.analyse(img, finder.find(img)[:1]) if img is not None else {"faces": []}
        if res["faces"]:
            ys.append(label)
            ps.append(res["faces"][0]["p_ai"])
    ys, ps = np.array(ys), np.array(ps)
    pred = ps >= 0.5
    return {"accuracy": float((pred == (ys == 1)).mean()), "ai_recall": float(pred[ys == 1].mean()),
            "real_specificity": float((~pred[ys == 0]).mean()), "n": int(len(ys))}


def train(real_dir: Path, ai_dir: Path, epochs: int, out: Path, batch: int = 64, workers: int = 4, limit: int = 0) -> Dict:
    """Train, keep the best checkpoint by stress accuracy, and return the per-epoch history."""
    import torch
    from torch.utils.data import DataLoader

    torch.manual_seed(0)
    r_tr, r_va = split(list_files(real_dir))
    a_tr, a_va = split(list_files(ai_dir))
    if limit:
        r_tr, r_va, a_tr, a_va = r_tr[:limit], r_va[:max(8, limit // 4)], a_tr[:limit], a_va[:max(8, limit // 4)]
    boxes = find_boxes(r_tr + r_va + a_tr + a_va)
    for name, group in (("real", r_tr + r_va), ("ai", a_tr + a_va)):
        logger.info("faces found in %d of %d %s images (images without a face are skipped)", sum(str(p) in boxes for p in group), len(group), name)
    keep = lambda ps: [p for p in ps if str(p) in boxes]
    r_tr, r_va, a_tr, a_va = keep(r_tr), keep(r_va), keep(a_tr), keep(a_va)
    tr = [(p, 0) for p in r_tr] + [(p, 1) for p in a_tr]      # label 1 = AI-generated
    va = [(p, 0) for p in r_va] + [(p, 1) for p in a_va]
    logger.info("train %d (real %d, ai %d) | validation %d (real %d, ai %d)", len(tr), len(r_tr), len(a_tr), len(va), len(r_va), len(a_va))

    model = build_model(pretrained=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(FaceSet(tr, boxes, True), batch_size=batch, shuffle=True, num_workers=workers, drop_last=True, persistent_workers=workers > 0)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1e-3, total_steps=epochs * len(loader))
    loss_fn = torch.nn.CrossEntropyLoss()
    best, history = -1.0, []
    for epoch in range(epochs):
        model.train()
        t0, total = time.time(), 0.0
        for x, y in loader:
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())
        clean = evaluate(model, FaceSet(va, boxes, False), workers=workers)
        stress = evaluate(model, FaceSet(va, boxes, False, stress=True), workers=workers)
        row = {"epoch": epoch + 1, "train_loss": round(total / max(1, len(loader)), 4), "val_clean": clean, "val_stress": stress, "seconds": round(time.time() - t0)}
        history.append(row)
        logger.info("%s", json.dumps(row))
        if stress["accuracy"] > best:
            best = stress["accuracy"]
            out.parent.mkdir(parents=True, exist_ok=True)
            meta = {"epochs_run": epoch + 1, "val_stress": stress, "val_clean": clean, "train_samples": len(tr), "val_samples": len(va),
                    "input_size": INPUT_SIZE, "label_1": "ai_generated"}
            torch.save({"model_state_dict": model.state_dict(), "meta": meta}, out)
    final = evaluate_inference(out, va)
    logger.info("inference-path validation (detector -> crop -> classifier, no degradation): %s", json.dumps(final))
    return {"best_stress_accuracy": best, "inference_path": final, "history": history}


def main(argv=None) -> Dict:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--real", type=Path, required=True)
    ap.add_argument("--ai", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--out", type=Path, default=CHECKPOINT)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap files per class")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    report = train(a.real, a.ai, a.epochs, a.out, workers=a.workers, limit=a.limit)
    print(json.dumps({"best_stress_accuracy": report["best_stress_accuracy"]}))
    return report


if __name__ == "__main__":
    main()

"""
services.build_vocab_embeddings: recompute the zero-shot vocabulary embeddings shipped in ``core/perception/vocab_embeddings.npz``.

    python -m services.build_vocab_embeddings

Run it after changing a vocabulary in ``core/perception/vocab.py`` or the pinned SigLIP 2 revision. The file stores one L2-normalised
text embedding per prompt plus a fingerprint of the model revision and prompts; the recognizer ignores a file whose fingerprint does
not match and embeds the prompts itself (slower), and a test fails until the file is rebuilt.
"""
from __future__ import annotations

import sys

import numpy as np

from core.perception import recognizer
from core.perception.vocab import VOCABS


def main() -> int:
    """Embed every vocabulary with the full model and write the npz; returns a process exit code."""
    rec = recognizer.ZeroShotRecognizer()
    rec._embed_vocabularies_and_free_text_tower = lambda *a, **k: None          # keep the text tower; we need it here
    if rec._ensure() is None:
        print("The SigLIP 2 model is not available; run `python -m services.fetch_models` first.", file=sys.stderr)
        return 1
    arrays = {vocab: rec._text_features(vocab).numpy().astype(np.float32) for vocab in VOCABS}
    np.savez_compressed(recognizer.EMBEDDINGS_FILE, key=np.array(recognizer.vocabulary_key()), **arrays)
    print(f"wrote {recognizer.EMBEDDINGS_FILE} ({sum(a.shape[0] for a in arrays.values())} prompts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

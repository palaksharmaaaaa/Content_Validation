# Limitations

What this project does not do, or has not been shown to do.

## Detection quality

- **Accuracy is unmeasured.** No labelled dataset ships; thresholds were tuned on synthetic fixtures. Use `services.calibration_cli` on your own labelled media to get real numbers ([how](USER_GUIDE.md#measure-real-accuracy)).
- **Heuristics, not a classifier.** Apart from the optional image backbone you train yourself, every AI-versus-real decision is hand-written threshold logic over pixel, spectral and metadata signals.
- **"PRNU" is a median-filter noise residual**, not a camera fingerprint.
- **Faces are found with YuNet**, a published CNN detector bundled in `core/models/`. Counts looked right on a manual review of 36 personal photos (including one 33 px background face), but recall and precision on your media are not measured. The facial *deepfake-risk* score on top of it is still a texture heuristic.
- **Attribution is a guess.** Only some generators have spectral calibration; the rest match metadata strings. Attribution never changes the verdict.
- **C2PA is presence-only**; signatures and hash bindings are not verified. **EXIF is unauthenticated.**
- **Learning is shallow.** Feedback nudges a few scalar weights; real model training is a separate, manual step. Near-duplicate files (re-saves, resizes) are not recognised as the same sample.

## Specification versus implementation

The three `GLOBAL_*_TAXONOMY_AND_FORENSIC_RESEARCH_REPORT.md` files describe a very broad target. Each ends with an implementation-status appendix. Dimensions marked spec-only (for example astrophysical, subatomic or event-sensor imagery, or zero-knowledge media proofs) are research scope: recognised or documented but not scored, and no claim of coverage is made for them.

## Operations

- **Not tested on GPU.** The CUDA code path exists but has never been run; the tests cover CPU behaviour.
- **CI has never run on GitHub.** The workflow is written and its commands pass locally.
- **No authentication.** Per-session scratch folders keep users' files apart, but the app is for personal, local use only.
- **Batch analysis is sequential.** The pipelines share models and were not designed for parallel runs.
- **Size limits:** image 100 MB, audio 200 MB or 1 hour, video 500 MB.

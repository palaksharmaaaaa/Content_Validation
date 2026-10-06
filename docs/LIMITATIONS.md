# Limitations

What this project does not do, or has not been shown to do.

## Detection quality

- **Accuracy is unmeasured.** No labelled dataset ships; thresholds were tuned on synthetic fixtures. Use `services.calibration_cli` on your own labelled media to get real numbers ([how](USER_GUIDE.md#measure-real-accuracy)).
- **Heuristics, not a classifier.** Apart from the optional image backbone you train yourself, every AI-versus-real decision is hand-written threshold logic over pixel, spectral and metadata signals.
- **"PRNU" is a median-filter noise residual**, not a camera fingerprint.
- **Faces are found with YuNet**, a published CNN detector bundled in `core/models/`. Counts looked right on a manual review of 36 personal photos (including one 33 px background face), but recall and precision on your media are not measured. The facial *deepfake-risk* score on top of it is still a texture heuristic.
- **The face-authenticity model is only as general as its training set.** It was trained on one Kaggle dataset (about 5,000 real faces, 4,600 AI faces) and scores 100 % on that dataset's held-out images, which says the dataset is easy, not that the model is. How it behaves on faces from other generators (Midjourney, Flux, face swaps, beautify filters) is unmeasured. On 44 faces from 36 real phone photos it called none AI-like, which checks false alarms only. The first version of this model learned the dataset's framing instead of faces and called every AI face real when run through the real pipeline; it was retrained on crops built exactly as at inference, and the training report now includes an inference-path accuracy.
- **Recognition is zero-shot.** Place, species and vehicle type are the best match inside a fixed vocabulary, so they always return something; results below 30 % are shown as "not sure". Rare species and specific vehicle models are weak spots.
- **Expression recognition is indicative.** The model reaches about 88 % on a posed-face benchmark; real photos (stern group poses, sunglasses, side views) are harder, and "disgust" in particular is over-reported.
- **Same-person matching is within one photo set only**: nothing is stored, no names are attached. It is accurate on frontal faces and weak on small, blurred or turned faces.
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

## Age and minor screening (`core/perception/age.py`)
- **No model finds every child.** MiVOLO v2 (Apache-2.0) is the strongest open age model found in the October 2026 research, with a mean error of about 3.7 years on its own benchmark and 4-5 years elsewhere, but a 2026 independent benchmark of 34 models found that off-the-shelf age models mislabel many minors as adults when the cut-off is exactly 18, and all of them are weakest under about 5 years old. The screening is therefore recall-first and sends uncertain people to a human (`review_required`).
- **The margin was measured, and it is a trade-off.** On 2,250 UTKFace faces with known ages (`python -m services.age_eval`), flagging "estimated age below 27, or the age-group opinion at least 50 % child/teenager" sends 99.2 % of under-18s to review (97.7 % of 16-17 year-olds) but also 87 % of 18-20, 75 % of 21-25 and 26 % of 26-35 year-olds. A plain "age < 18" rule finds only 85.6 % of minors (46.7 % of 16-17). Those faces are pre-cropped and well lit, with one face per image and no body; recall on real photographs, with small faces, occlusion, filters and make-up, will be lower and has not been measured. UTKFace itself is licensed for non-commercial research only.
- **Faces and bodies must be visible.** People whose face and body are both too small, cut off or hidden come back `UNDETERMINED`. Drawings, cartoons and AI-generated people are aged by appearance, which was not validated for them.
- **Apparent age, not real age.** Make-up, filters, lighting and image quality move the estimate; the result says nothing about a person's actual age or identity.
- **Weights.** `core/models/mivolo_v2/model.safetensors` is the upstream file at revision `53393526c220e34cdd7b722b36d22b6f9e5f4241` (115 MB, stored with Git LFS). The network code in `core/perception/mivolo_vendor/` is vendored from the MiVOLO repository (Apache-2.0, licence included).

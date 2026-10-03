# Verification of the old "Brutal Audit" (BRUTAL_AUDIT_REPORT.md, 2026-10-02)

Every claim was re-checked against the code as it stands today. Regression tests live in `tests/test_audit_regressions.py`
(plus `core/tests/test_secure_fetch.py` and `tests/test_app_end_to_end.py`).

Legend: **Fixed now** = valid, fixed in this change · **Already fixed** = valid when written, fixed earlier · **Not a bug** =
claim does not hold · **Open** = valid, deliberately not changed (reason given).

## Critical
| ID | Verdict | Notes |
|---|---|---|
| C-1 DNS-rebinding SSRF | Already fixed | `_PinnedResolver` pins each request to the validated IPs; redirects re-validated per hop. |
| C-2 IPv4/IPv6 `TypeError` | **Not a bug** | `IPv4Address in IPv6Network` returns `False`; it does not raise (checked on this Python). |
| C-3 Fake C2PA | Already fixed | Presence-only, never scored, never labelled "verified" (provenance view, decision layer, UI, narratives). |
| C-4 `NameError` in feedback UI | **Fixed now** | `IPTC_SOURCE_TYPE_MAPPING` was used but never imported; crashed when no nine-dimension dossier existed. |
| C-5 Audio checkpoint key | **Fixed now** | Trainer writes `model_state_dict`; loader looked for `state_dict`. Loader accepts both. |
| C-6 fd leak on download | Already fixed | Temp file is created only for an accepted download; tests assert no leftovers. |

## High
| ID | Verdict | Notes |
|---|---|---|
| H-1 Skin-colour "face detection" | **Open** | Valid. Replacing it with a Haar/DNN detector cannot be validated here (no labelled face images, no network); a blind swap could lower recall on real faces. Do it together with the calibration tool once labelled data exists. |
| H-2 Vocoder cutoff false positives | **Fixed now** | A cutoff within 8 % of Nyquist is ordinary anti-aliasing and no longer counts. |
| H-3 Undecided-margin scale | **Fixed now** | Gap was on a 0-100 scale but clamped as 0-1, so the margin was always the 3 % floor. |
| H-4 Fusion overwritten by image result | Already fixed | `decision_mode` (`image_authoritative` vs `fused`); video verdict comes from the video detector. |
| H-5/H-6 Model-load races | **Fixed now** | Locks around the shared vision model and all three detector `load()` methods. |
| H-7 Learner lost updates | **Fixed now** | `record_feedback` holds the per-file locks for the whole read-modify-write. |
| H-8 Unbounded lock registry | **Fixed now** | Weak-valued registry. |
| H-9 Linear-interpolation resampling | **Fixed now** | Windowed-sinc low-pass before downsampling. |
| H-10 Video provenance reads 64 KB only | **Fixed now** | Atom names are searched in head and tail (vendor strings already were). |
| H-11 Correlation-discount key mismatch | Already fixed | Keys now match (`sensor_noise` / `surface_smoothness`). |
| H-12 3xx accepted after redirect budget | **Fixed now** | Exhausted budget is an error. |

## Medium
| ID | Verdict | Notes |
|---|---|---|
| M-1 Session cache collisions | Already fixed | Per-session scratch directories; wipe is session-scoped. |
| M-2 Triple copy-paste in `app.py` | **Fixed now** | One `ui/media_tab.py` driven by a spec; `app.py` 497 -> ~230 lines. |
| M-3 Name+size cache signature | **Fixed now** | Content hash; end-to-end test uploads same-name files with different bytes. |
| M-4 One lock for all lazy properties | **Fixed now** | Per-component locks. |
| M-5 Object detector CPU-only | **Fixed now** | Uses CUDA when present (CPU path unchanged and tested; GPU path not exercised here). |
| M-6 "High-speed" sequential batch | **Fixed now (docs)** | Docstrings now say sequential. Parallelism not added: the pipelines share models and were not designed for it. |
| M-7 Unclosed `VideoCapture` | Not applicable | The code moved into the pipeline; all extractor captures release in `finally`. |
| M-8 Negative duration | **Fixed now** | Frame count clamped. |
| M-9 `cleanup_url_download` never called | **Fixed now** | URL downloads go into the session scratch dir, removed by the wipe button. |
| M-10 `fsync` on UI thread | **Fixed now** | Removed. |
| M-11 "suno" key mismatch | **Not a bug** | Code uses `suno_ai`. **But** verifying it exposed a real bug: `"udio" in text` matched inside "audio", attributing any file with such metadata to Udio. Now whole-word matching. |
| M-12 Confidence paradox | **Not reproduced** | More cues for the same generator raise its share (0.60 -> 0.81); it falls only when a competitor also gains evidence. |
| M-13 Relative redirects | **Fixed now** | `urljoin` before validation. |

## Low
L-2 (docstring claims), L-3 (dead branch), L-4 (crest factor vs dynamic range: now both, correctly named), L-5 (mid-file imports),
L-6 (dead `pass`), L-7 (dead key), L-8/L-9 (silent exceptions: now logged), L-10 (duplicated feedback panel) are **fixed**.
L-1 / L-11 (audio and video dossiers reuse image-style key names such as `dimension_3`) are **open**: the keys are a
public contract of the dossier consumed by the UI; titles/labels are correct, so renaming was not worth the breakage.

## Architecture / ML section
- "No pretrained checkpoint is shipped; 100 % heuristics on a fresh install": **true, and now deliberate** (the project ships blank).
- "PRNU is a median-filter residual, not real PRNU": **true**; documented as a heuristic.
- "Test coverage zero for detection paths": **no longer true** (seeded golden tests for faces, ELA/metadata, screenshots, art,
  taxonomy, attribution, dossiers, pipelines, rendered UI, end-to-end app).
- CI/CD: workflow added (`.github/workflows/tests.yml`, not yet run on GitHub). Dependencies: exact versions in `requirements.lock.txt`.
- "Pervasive `except Exception: pass`": reduced where findings pointed (hashing, attribution); many broad handlers remain by design in
  feature extractors that must never crash the pipeline.

# Audio Dimension Checks — Design Spec

**Date:** 2026-10-02
**Builds on:** `2026-10-02-dimension-checks-foundation-image-design.md` (all shared decisions carry over: Finding model, hybrid capped scoring, hard-block hash gate, recognition-only, advisory channels, five bands, OOD gate, `ui/stages.py`).
**Scope:** wire the audio modality into the foundation. Video is a later cycle.

## 1. Decisions specific to audio

| Topic | Decision |
|---|---|
| Score integration | `AudioAIDetector.analyze_audio_file(..., extra_log_lrs=None)`. Audio pools by weighted average (not log-odds), so after pooling `p_ai` is shifted in base-10 log-odds space: `p' = sigmoid10(logit10(p) + sum(terms))`, re-clipped to [0.01, 0.99]. Empty/None terms leave results byte-identical. Same caps as image (per finding 0.25, explicit generator string 0.40, total ±0.40). |
| Evidence that may score | Self-declared generator strings in tags (+0.40, explicit cap), ENF continuous trace (−0.15, PHYSICAL_SIGNAL), ENF splice discontinuity (+0.15). Everything else is advisory. Absence (e.g. no ENF) never scores. |
| Decode | Existing 16 kHz mono PCM is reused for ENF/telephony/loudness. Native-rate content (fake hi-res, bit depth) is read from WAV via `wave`, or via ffmpeg (native rate, ≤30 s); if neither is available the check is NOT_APPLICABLE. |
| Dependencies | None new (numpy, stdlib, optional ffmpeg already required by the project). |
| Shared code | Byte-scan helpers (signatures, windows, injection patterns) move to `core/forensics/bytescan.py`; image integrity re-uses them. |
| Recognition | Symbolic music (MIDI `MThd`) is recognized as outside waveform-authenticity scope. |
| Commits | Not made automatically. |

## 2. Checks

`audio_detector/dimension_checks/`: `_common.py`, `integrity.py`, `container.py`, `signal.py`, `context.py`, `legal.py`, `lifecycle.py`, `reliability.py`, `fit_ood.py`, `__init__.py` (`AudioDimensionAnalysis`, `check_audio_gates`, `gate_short_circuit_result`, `summarize_for_evidence_trail`).

| Stage | check_id | Class | llr |
|---|---|---|---|
| file_integrity | `format_sniff` (RIFF/WAVE, ID3/MPEG sync, fLaC, OggS, ftyp, ADTS vs extension) | SECURITY | – |
| file_integrity | `trailing_data` (WAV RIFF size vs file, ID3v1/APE ignored) | SECURITY | – |
| file_integrity | `polyglot_signatures` (shared bytescan) | SECURITY | – |
| file_integrity | `prompt_injection_text` (ID3 text frames, RIFF INFO, Vorbis comments) | SECURITY | – |
| container | `id3_tags` (version/flags/size, frame inventory, APIC, padding; explicit generator strings in TXXX/COMM/TSSE/TENC) | METADATA_WEAK | +0.40 on explicit generator string |
| container | `riff_structure` (chunk walk, size consistency, `fact` for non-PCM, `bext` + CodingHistory, LIST INFO ISFT) | METADATA_WEAK | – |
| container | `flac_md5` (STREAMINFO MD5 vs ffmpeg 16-bit PCM MD5) | METADATA_WEAK | – |
| container | `mp3_encoder_tag` (Xing/Info + LAME version, lowpass) | METADATA_WEAK | – |
| container | `ogg_structure` (page CRC-32, serial changes, granule monotonicity) | METADATA_WEAK | – |
| signal | `enf_trace` (50/60 Hz mains trace; continuity/splice) | PHYSICAL_SIGNAL | −0.15 continuous, +0.15 splice, none if absent |
| signal | `fake_hires` (native-rate band-limit vs container rate; 16-in-24 bit padding; lossy-origin cutoff in lossless container) | PHYSICAL_SIGNAL | – |
| signal | `telephony_channel` (300–3400 Hz band-limiting) | PHYSICAL_SIGNAL | – |
| signal | `loudness_dynamics` (RMS/peak dBFS, crest factor, approx. true peak, clipping, over-compression) | PHYSICAL_SIGNAL | – |
| context (post) | `audio_fingerprint` (Haitsma–Kalker style 32-bit/frame fingerprint; optional local index) | CONTEXT | – |
| legal (post) | rights notice (ID3 TCOP/TPUB, RIFF ICOP, bext Originator), voice-biometric notice (speech-dominant), synthetic-speech likeness/disclosure advisory | LEGAL_FLAG | – |
| lifecycle (post) | `transcoding_cascade` | RELIABILITY | – |
| reliability (post) | `confidence_limiters` (telephony band, low bitrate, very short, heavy clipping, probable cascade) | RELIABILITY | – |

ENF method: 16 kHz mono → decimate to 1 kHz → 16 s Hann STFT windows, 1 s hop → peak within ±1 Hz of 50 and of 60 → parabolic interpolation → per-frame SNR (peak vs median in band). Present if median SNR ≥ 4 over ≥ 60% of frames and ≥ 10 s; splice if a single inter-frame jump > 0.15 Hz occurs between two high-SNR frames. Absent ⇒ INFO only ("battery power, telephony, noise reduction and many studios remove mains hum; this is not evidence of synthesis").

## 3. Gates, bands, OOD, open set

- Gates: hard-block SHA-256 (shared), symbolic-music recognition (`MThd`).
- Band from `ai_percentage`; OOD gate on the 5-dim acoustic feature vector (`has_vocoder_cutoff`, `cutoff_hz/10000`, `flatness*100`, `digital_silence_ratio`, `high_freq_ratio`), fitted via `python -m audio_detector.dimension_checks.fit_ood` from the audio media library; `NOT_CALIBRATED` until then.
- `UNKNOWN_SOURCE` when attribution `model_key` is unknown/`Unknown` and `ai_percentage ≥ 60`.

## 4. Integration

Both `AudioForensicPipeline.analyze` and `ui.batch_ui.process_single_audio` use the same helpers: gates → (validation) → `AudioDimensionAnalysis.run_pre()` (→ `extra_log_lrs`) → detector → scene/attribution → `run_post()`. Result adds `dimension_report`, `confidence_band`, `ood`, `attribution_open_set`, `gate`. UI: `render_linear_audio_pipeline_results` gains the gate banner and Stages 1b (File Integrity & Security), 3b (Container, Metadata & Signal Forensics), 4b (Confidence Band & Reliability), 6b (Context, Rights & Lifecycle), reusing `ui/stages.py`.

## 5. Testing

TDD with synthetic fixtures (numpy-generated WAVs, hand-built ID3/RIFF/Ogg/FLAC bytes; ffmpeg-dependent tests skip if ffmpeg is absent). Cap/identity regression: `analyze_audio_file` unchanged with no terms. Full suite stays green. AppTest render of the audio flow.

## 6. Docs

Audio report Appendix F updated; README section; spec/plan retained in `docs/superpowers/`.

## 7. Out of scope

M4A/AAC atom forensics, true BS.1770 LUFS, formant tracking/RT60, watermark detectors (AudioSeal/WavMark/SynthID), speaker diarization/LID, ASR-based cheapfake checks, video.

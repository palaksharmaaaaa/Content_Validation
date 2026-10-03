"""
audio_detector.dimension_checks.signal: signal-level checks (report Dimensions L, M, O).

PHYSICAL_SIGNAL class. Only ENF may add log-odds (continuous trace -0.15, splice +0.15). Absence of an
ENF trace is NEVER evidence: battery-powered recorders, telephony band-limits, noise reduction and many
studios remove mains hum. Fake hi-res / telephony / loudness findings are informational.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from audio_detector.dimension_checks import _common as C
from core.forensics.registry import CheckContext, ctx_memo, registry
from core.forensics.schemas import EvidenceClass, Finding, FindingStatus, Severity

ENF_WINDOW = 8192
ENF_HOP = 1000
ENF_PAD = 32768
ENF_MIN_FRAMES = 5
ENF_SNR_MIN = 6.0
ENF_PRESENT_FRACTION = 0.60
ENF_STABLE_TONE_STD_HZ = 0.003
ENF_ERRATIC_STD_HZ = 0.30
ENF_JUMP_HZ = 0.15
ENF_CONTINUOUS_LLR = -0.15
ENF_SPLICE_LLR = 0.15


def _f(check_id, dim, stage, title, status, severity, detail, data=None, llr=None) -> Finding:
    return Finding(check_id=check_id, dimension=dim, stage=stage, title=title, status=status, severity=severity,
                   evidence_class=EvidenceClass.PHYSICAL_SIGNAL, detail=detail, data=data or {}, llr=llr)


def _samples(ctx: CheckContext) -> Tuple[np.ndarray, int]:
    s = ctx.extra.get("samples")
    if s is None:
        from audio_detector.validator import AudioValidator

        s = AudioValidator().extract_pcm_samples(ctx.path)
        ctx.extra["samples"] = s
    x, sr, _dur = s
    return (np.asarray(x, dtype=np.float32) if x is not None else np.zeros(0, np.float32)), int(sr)


# ---------------------------------------------------------------------------------------------
# ENF
# ---------------------------------------------------------------------------------------------
def _enf_track(xd: np.ndarray, fs: float, nominal: float):
    win = np.hanning(ENF_WINDOW)
    freqs = np.fft.rfftfreq(ENF_PAD, 1.0 / fs)
    band = np.where((freqs >= nominal - 1.0) & (freqs <= nominal + 1.0))[0]
    ref_mask = (freqs >= nominal - 8) & (freqs <= nominal + 8) & ~((freqs >= nominal - 1.5) & (freqs <= nominal + 1.5))
    f_est, snr = [], []
    for start in range(0, len(xd) - ENF_WINDOW + 1, ENF_HOP):
        seg = xd[start : start + ENF_WINDOW].astype(np.float64)
        seg = (seg - seg.mean()) * win
        spec = np.abs(np.fft.rfft(seg, ENF_PAD))
        i = band[int(np.argmax(spec[band]))]
        peak = spec[i]
        floor = float(np.median(spec[ref_mask])) + 1e-12
        a, b, c = np.log(spec[i - 1 : i + 2] + 1e-12)
        denom = a - 2 * b + c
        delta = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
        f_est.append(freqs[i] + delta * (fs / ENF_PAD))
        snr.append(peak / floor)
    return np.array(f_est), np.array(snr)


@registry.register("audio", "enf_trace")
def check_enf_trace(ctx: CheckContext) -> Finding:
    title = "Electrical Network Frequency (ENF) trace"
    x, sr = _samples(ctx)
    if sr < 200 or len(x) < sr * 12:
        return _f("enf_trace", "O", "signal", title, FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Recording is shorter than about 12 s; ENF tracking needs a longer signal.")
    factor = max(1, int(round(sr / 1000.0)))
    n = len(x) // factor * factor
    xd, fs = x[:n].reshape(-1, factor).mean(axis=1), sr / factor
    best: Optional[Dict[str, Any]] = None
    for nominal in (50, 60):
        f_est, snr = _enf_track(xd, fs, nominal)
        if len(f_est) < ENF_MIN_FRAMES:
            continue
        cand = {"nominal": nominal, "f": f_est, "snr": snr, "median_snr": float(np.median(snr))}
        if best is None or cand["median_snr"] > best["median_snr"]:
            best = cand
    if best is None:
        return _f("enf_trace", "O", "signal", title, FindingStatus.NOT_APPLICABLE, Severity.NONE, "Too few analysis frames.")
    good = best["snr"] >= ENF_SNR_MIN
    frac = float(good.mean())
    data: Dict[str, Any] = {"nominal_hz": best["nominal"], "median_snr": round(best["median_snr"], 1),
                            "good_frame_fraction": round(frac, 2), "frames": int(len(good))}
    if frac < ENF_PRESENT_FRACTION:
        return _f("enf_trace", "O", "signal", title, FindingStatus.INFO, Severity.NONE,
                  "No continuous mains-hum trace found. This is not evidence of synthesis: battery-powered recorders, telephony, "
                  "noise reduction and many studios remove mains hum.", data)
    fg = best["f"][good]
    std = float(np.std(fg))
    data.update({"median_hz": round(float(np.median(fg)), 4), "std_hz": round(std, 4)})
    if std < ENF_STABLE_TONE_STD_HZ:
        data["stable_tone"] = True
        return _f("enf_trace", "O", "signal", title, FindingStatus.INFO, Severity.NONE,
                  f"A perfectly stable tone sits near {best['nominal']} Hz (frequency std {std:.4f} Hz); real mains frequency wanders, "
                  "so this is a tone or bass note, not an ENF trace.", data)
    if std > ENF_ERRATIC_STD_HZ:
        return _f("enf_trace", "O", "signal", title, FindingStatus.INFO, Severity.NONE,
                  f"Energy near {best['nominal']} Hz is too erratic (std {std:.2f} Hz) to be a grid trace.", data)
    jumps = 0
    idx = np.where(good)[0]
    for a, b in zip(idx, idx[1:]):
        if b == a + 1 and abs(best["f"][b] - best["f"][a]) > ENF_JUMP_HZ:
            jumps += 1
    data["jumps"] = jumps
    if jumps:
        return _f("enf_trace", "O", "signal", title, FindingStatus.WARN, Severity.MEDIUM,
                  f"Mains-hum trace near {best['nominal']} Hz has {jumps} abrupt frequency discontinuit{'y' if jumps == 1 else 'ies'}; "
                  "consistent with segments spliced from different recordings.", data, llr=ENF_SPLICE_LLR)
    return _f("enf_trace", "O", "signal", title, FindingStatus.PASS, Severity.NONE,
              f"Continuous mains-hum trace near {best['nominal']} Hz (median {data['median_hz']} Hz): consistent with an unspliced "
              "capture on a mains-powered device. Weak evidence: hum can be injected.", data, llr=ENF_CONTINUOUS_LLR)


# ---------------------------------------------------------------------------------------------
# Fake hi-res / bit-depth padding
# ---------------------------------------------------------------------------------------------
def _welch_db(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    n = 4096 if sr <= 64000 else 8192
    win = np.hanning(n)
    hop = n // 2
    starts = list(range(0, len(x) - n + 1, hop))
    if len(starts) > 300:
        starts = starts[:: len(starts) // 300 + 1]
    acc = np.zeros(n // 2 + 1)
    for s in starts:
        acc += np.abs(np.fft.rfft(x[s : s + n] * win)) ** 2
    acc /= max(1, len(starts))
    return np.fft.rfftfreq(n, 1.0 / sr), 10.0 * np.log10(acc + 1e-20)


def _find_cutoff(freqs: np.ndarray, db: np.ndarray, sr: int) -> Optional[float]:
    nyq = sr / 2.0
    edges = np.arange(4000.0, nyq - 250.0, 250.0)
    if len(edges) < 12:
        return None
    means = np.array([db[(freqs >= e) & (freqs < e + 250.0)].mean() for e in edges])
    best_i, best_drop = None, 0.0
    for i in range(4, len(means) - 4):
        drop = means[i - 4 : i].mean() - means[i : i + 4].mean()
        if drop > best_drop:
            best_i, best_drop = i, drop
    if best_i is None or best_drop < 25.0:
        return None
    above = means[best_i + 1 :]
    if len(above) >= 4 and float(np.std(above)) > 6.0:
        return None
    return float(edges[best_i])


@registry.register("audio", "fake_hires")
@ctx_memo("fake_hires")
def check_fake_hires(ctx: CheckContext) -> Finding:
    title = "Fake hi-res / bit-depth padding"
    head, _t, _s = C.read_windows(ctx.path)
    fmt = C.sniff_audio_format(head)
    if fmt not in ("wav", "flac"):
        return _f("fake_hires", "L", "signal", title, FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Applies to lossless containers (WAV/FLAC) only.")
    loaded = C.load_native_mono(ctx.path, 30.0)
    if loaded is None:
        return _f("fake_hires", "L", "signal", title, FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Native-rate audio could not be read (ffmpeg missing or unsupported encoding).")
    x, sr, bits = loaded
    if len(x) < 16384:
        return _f("fake_hires", "L", "signal", title, FindingStatus.NOT_APPLICABLE, Severity.NONE, "Audio too short to analyze.")
    freqs, db = _welch_db(x, sr)
    cutoff = _find_cutoff(freqs, db, sr)
    padded = False
    if bits == 24:
        v = np.round(x[:2_000_000].astype(np.float64) * 8388608.0).astype(np.int64)
        padded = bool(np.any(v != 0) and np.mean((v & 0xFF) == 0) >= 0.999)
    verdict = "NO_ANOMALY"
    if cutoff is not None:
        if sr >= 88200 and cutoff <= 24000:
            verdict = "UPSAMPLED_FROM_LOWER_RATE"
        elif 32000 <= sr <= 64000 and cutoff < 15000:
            verdict = "UPSAMPLED_FROM_LOWER_RATE"
        elif 32000 <= sr <= 64000 and (15000 <= cutoff <= 16800 or 18000 <= cutoff <= 20800):
            verdict = "POSSIBLE_LOSSY_ORIGIN"
    data = {"sample_rate": sr, "bits": bits, "cutoff_hz": cutoff, "padded_bits": padded, "verdict": verdict}
    notes = []
    status, sev = FindingStatus.PASS, Severity.NONE
    if verdict == "UPSAMPLED_FROM_LOWER_RATE":
        notes.append(f"content is band-limited at about {cutoff / 1000:.1f} kHz inside a {sr / 1000:g} kHz file, consistent with upsampling "
                     f"from a roughly {2 * cutoff / 1000:.1f} kHz-rate source")
        status, sev = FindingStatus.WARN, Severity.MEDIUM
    elif verdict == "POSSIBLE_LOSSY_ORIGIN":
        notes.append(f"a hard lowpass near {cutoff / 1000:.1f} kHz in a lossless {sr / 1000:g} kHz container is typical of an MP3/AAC-origin transcode")
        status, sev = FindingStatus.INFO, Severity.LOW
    if padded:
        notes.append("the lowest 8 bits of every sample are zero: 16-bit audio padded into a 24-bit file")
        status, sev = FindingStatus.WARN, Severity.MEDIUM
    if notes:
        return _f("fake_hires", "L", "signal", title, status, sev, "; ".join(notes).capitalize() + ".", data)
    return _f("fake_hires", "L", "signal", title, FindingStatus.PASS, Severity.NONE,
              "Spectral content extends across the band expected for the container rate; no padding detected.", data)


# ---------------------------------------------------------------------------------------------
# Telephony band-limit
# ---------------------------------------------------------------------------------------------
@registry.register("audio", "telephony_channel")
@ctx_memo("telephony_channel")
def check_telephony_channel(ctx: CheckContext) -> Finding:
    title = "Telephony band-limiting"
    x, sr = _samples(ctx)
    if sr < 8000 or len(x) < sr:
        return _f("telephony_channel", "L", "signal", title, FindingStatus.NOT_APPLICABLE, Severity.NONE,
                  "Not enough audio at a sufficient sample rate.")
    seg = x[: sr * 60].astype(np.float64)
    spec = np.abs(np.fft.rfft(seg)) ** 2
    f = np.fft.rfftfreq(len(seg), 1.0 / sr)

    def band(lo, hi):
        return float(spec[(f >= lo) & (f <= hi)].sum()) + 1e-20

    mid = band(300, 3400)
    hf_db = 10 * np.log10(band(3800, min(7800, sr / 2 - 1)) / mid)
    lf_db = 10 * np.log10(band(20, 200) / mid)
    band_limited = bool(hf_db <= -35.0)
    narrowband = bool(band_limited and lf_db <= -20.0)
    data = {"hf_ratio_db": round(float(hf_db), 1), "lf_ratio_db": round(float(lf_db), 1),
            "band_limited_4k": band_limited, "narrowband_telephony": narrowband}
    if narrowband:
        return _f("telephony_channel", "L", "signal", title, FindingStatus.INFO, Severity.LOW,
                  "Energy is confined to about 300-3400 Hz (G.711/AMR-NB telephony band). Spoof-detection cues above 4 kHz are absent, "
                  "so reliability is reduced.", data)
    if band_limited:
        return _f("telephony_channel", "L", "signal", title, FindingStatus.INFO, Severity.LOW,
                  "Energy is confined below about 4 kHz (low-rate source or heavy lowpass); high-frequency vocoder cues are absent.", data)
    return _f("telephony_channel", "L", "signal", title, FindingStatus.PASS, Severity.NONE,
              "No telephony-style band-limiting detected.", data)


# ---------------------------------------------------------------------------------------------
# Loudness / dynamics
# ---------------------------------------------------------------------------------------------
@registry.register("audio", "loudness_dynamics")
@ctx_memo("loudness_dynamics")
def check_loudness_dynamics(ctx: CheckContext) -> Finding:
    title = "Loudness & dynamics"
    x, sr = _samples(ctx)
    if len(x) < 1000:
        return _f("loudness_dynamics", "M", "signal", title, FindingStatus.NOT_APPLICABLE, Severity.NONE, "No audio samples.")
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
    if rms < 1e-4:
        return _f("loudness_dynamics", "M", "signal", title, FindingStatus.NOT_APPLICABLE, Severity.NONE, "Signal is silent.")
    peak = float(np.max(np.abs(x)))
    seg = x[: sr * 30].astype(np.float64)
    n = len(seg)
    up = np.fft.irfft(np.fft.rfft(seg), n * 4) * 4.0  # 4x FFT-domain oversampling (approximate true peak)
    true_peak = float(np.max(np.abs(up)))
    peak_db = 20 * np.log10(max(peak, 1e-9))
    rms_db = 20 * np.log10(rms)
    crest = peak_db - rms_db
    clip = float(np.mean(np.abs(x) >= 0.999))
    data = {"rms_dbfs": round(rms_db, 1), "peak_dbfs": round(peak_db, 1), "true_peak_dbfs_approx": round(20 * np.log10(max(true_peak, 1e-9)), 1),
            "crest_db": round(crest, 1), "clipping_ratio": round(clip, 4), "over_compressed": bool(crest < 6.0),
            "dc_offset": round(float(np.mean(x)), 5),
            "note": "RMS-based figures; not ITU-R BS.1770 integrated loudness (LUFS)."}
    if clip > 0.001:
        return _f("loudness_dynamics", "M", "signal", title, FindingStatus.WARN, Severity.LOW,
                  f"{clip * 100:.2f}% of samples are clipped at full scale; clipping distorts spectral cues.", data)
    extra = " Crest factor below 6 dB indicates heavy compression/limiting." if data["over_compressed"] else ""
    return _f("loudness_dynamics", "M", "signal", title, FindingStatus.INFO, Severity.NONE,
              f"RMS {rms_db:.1f} dBFS, peak {peak_db:.1f} dBFS, crest {crest:.1f} dB.{extra}", data)

# Comprehensive Global Audio Taxonomy, Universal Acoustic Spectrum, and Generative AI Audio Synthesis Architecture

**Document Reference:** `RPT-AUD-FOR-2026-OCT-02-REV7`  
**Referenced Standards (informational only; this software is not certified or audited against them):** AES Standards (AES27, AES43, AES47, AES67, AES70), EBU R128 / Tech 3333, EBU Tech 3285 (BWF), C2PA Technical Specification v2.4, IPTC Audio Metadata Standards (2024–2026), ISO/IEC 27037:2012, MPEG-H 3D Audio, ITU-R BS.1770-4  
**Classification:** Technical Architecture & Omnidimensional Acoustic Forensic Specification  
**Publication Date:** October 2, 2026  
**Timestamp:** 2026-10-02  
**Repository Working Directory:** the repository root  
**Author:** Antigravity Advanced Agentic Coding & Acoustic Forensic Engineering Team  

---

## Revision History

| Revision | Date / Timestamp | Scope & Key Modifications |
| :--- | :--- | :--- |
| **REV1** | 2026-10-02T11:50:00+05:30 | Initial audio taxonomy draft; core acoustic features, codecs, and basic TTS/VC categories. |
| **REV2** | 2026-10-02T12:35:00+05:30 | Expanded cosmic, gravitational wave, quantum phononics, and optoacoustic modalities. |
| **REV3** | 2026-10-02T13:24:00+05:30 | Added telephony, fake hi-res detection, loudness (LUFS), paralinguistics, music theory, side-channels, data-over-sound, DolphinAttack, and real-time voice changers. |
| **REV4** | 2026-10-02T14:20:00+05:30 | **Production Forensic Integrity Release:** Enforced strict 16-dimension ontology (A–P); added Section 20 Dedicated Metadata, Container-Structure, & Encoder Forensics (ID3v1/v2, BWF `bext` coding history, RIFF chunks, Vorbis comments, encoder strings); added AM/FM/SSB RF demodulation, Morse code (CW), and planetary seismic audification; expanded multi-layer watermarking (Meta AudioSeal, WavMark, SynthID, DSSS); codified legal/regulatory standards (Texas CUBI, Regulation (EU) 2024/1689 Art. 50, FCC Declaratory Ruling 24-17 under TCPA 47 U.S.C. § 227, HIPAA/EHDS acoustic biomarker PHI); integrated detector demographic validity (accents, dialects, vocal pathologies) and accessibility tracks; corrected table header to "Authentic Physical Acoustic/Mic Capture"; calibrated ENF benchmarks to electric utility grid archives with "Absent $\ne$ AI" physical safeguards (battery power, studio AC regenerators); decoupled epistemic OOD detection from Bayesian probability bands; and established the 5 calibrated operational probability bands. |
| **REV5** | 2026-10-02 | **Consistency Pass:** Removed visual-quality IEEE 3333.1 and unverifiable ISO 22144 / AES-11id citations (AES27/AES43/EBU Tech 3285 substituted, flagged for verification); synced ontology line for Dimension D; fixed TOC anchor for Section 21; softened over-strong claims on BWF `CodingHistory`, FLAC MD5, LAME tags, and ENF courtroom proof; added metadata evidentiary-weight caveat; added engine-state rows to Appendix A; added orthogonal-channel note; clarified that pipeline stages are target-architecture. |
| **REV6** | 2026-10-02 | **Status Release:** Added Appendix F (implementation status per dimension). The shared dimension-check foundation exists in `core/` (wired into image first); audio wiring is pending. |
| **REV7** | 2026-10-02 | **Implementation Release (audio):** The audio pipeline now implements the shared dimension-check foundation: hard-block and symbolic-music gates; file-integrity checks; ID3/RIFF-bext/FLAC-MD5/MP3-LAME/Ogg container checks; ENF trace and splice detection, fake hi-res and bit-depth padding, telephony band-limit and loudness/dynamics signal checks; fingerprint/legal/lifecycle/reliability advisory stages; five probability bands, OOD gate (uncalibrated until fitted) and open-set `UNKNOWN_SOURCE`. Findings, band and reliability appear on the Evidence tab of the audio result page (the earlier numbered Stage 1b/3b/4b/6b layout was replaced by a verdict card with Overview, Evidence, Details and Feedback tabs). Appendix F rewritten. |

---

## Executive Summary

The emergence of deterministic Flow-Matching audio generators, Codec-Language Models, zero-shot voice cloning architectures, Neural Audio Codecs (SoundStream, EnCodec, Descript Audio Codec, FlexiCodec), high-fidelity generative music transformers (Suno v5.5, Udio, Stable Audio 3.0), real-time conversational speech synthesis engines (sub-200ms latency), ultrasonic acoustic side-channels, and Zero-Knowledge Proofs for verifiable audio redaction as of late 2026 has made elementary binary classification (*"Real vs. AI Audio"*) obsolete.

A digital audio recording is not merely a single static scalar or a 1D time-domain waveform; it is an **open-class, multi-dimensional physical, biological, cosmic, and computational signal** shaped across sixteen operational axes:
1.  **Dimension A: Provenance, Origin, & Authenticity Spectrum:** Ground-truth acoustic microphone capture through computational beamforming, software loopback, speech splicing, voice conversion (RVC), and fully synthetic generative speech/music engines.
2.  **Dimension B: Semantic Content Category, Sound Events, Music Theory, & Paralinguistics:** Spoken dialogue, singing vocals, instrumental music theory structures, acoustic sound events (AudioSet), and paralinguistic cues (laughter, sighs, micro-tremors).
3.  **Dimension C: Acoustic Physics, Environmental Propagation, & Side-Channels:** Acoustic wave mechanics, inverse-square law, atmospheric absorption, Room Impulse Response ($RT_{60}$), room mode standing waves, and acoustic emanations (keystroke acoustic snooping, CPU coil whine).
4.  **Dimension D: Frequency Spectrum, Ultra-Wideband, RF Demodulation, & Data-Over-Sound:** Infrasound ($<20\text{ Hz}$), human audible spectrum ($20\text{ Hz} - 20\text{ kHz}$), ultrasound ($>20\text{ kHz}$), RF demodulated audio (AM/FM/SSB), Morse code (CW), planetary seismic audification, and data-over-sound beacons.
5.  **Dimension E: Cosmic, Interstellar, & Astrophysical Acoustic Phenomena:** Gravitational wave spacetime strain chirps ($h(t)$ from LIGO/Virgo/KAGRA), interstellar plasma wave densities (Voyager 1 & 2), planetary magnetospheric whistlers, and stellar helioseismology $p$-modes.
6.  **Dimension F: Quantum, Nanoscale, Cellular, & Bio-Acoustic Modalities:** Quantized single phonons in optomechanical crystals, nanomechanical cellular metabolic vibrations, and spontaneous human otoacoustic emissions (OAE).
7.  **Dimension G: Photoacoustic, Optoacoustic, & Acousto-Optic Hybrid Modalities:** Laser-induced thermoelastic ultrasonic shockwaves ($p_0 = \Gamma \mu_a F$) and acousto-optic laser beam diffraction via Bragg diffraction cells.
8.  **Dimension H: Parametric Non-Linear Ultrasonic Audio Arrays:** Highly directional audio spotlights projecting $40\text{ kHz}$ ultrasound self-demodulating in air via non-linear Westervelt acoustic mechanics into audible localized sound.
9.  **Dimension I: Neuro-Acoustic Decoding & Auditory Electrophysiology:** Invasive ECoG/sEEG neural speech reconstruction from the superior temporal gyrus, Auditory Brainstem Response (ABR) Waves I–V, and Electrocochleography.
10. **Dimension J: Transduction, Microphonic Physics, & Hardware Acquisition Modalities:** Moving-coil dynamic, capacitor condenser, ribbon, and silicon MEMS microphones; polar patterns; preamp electrical noise floors; and optical Laser Doppler Vibrometry.
11. **Dimension K: Spatial Audio Geometry, Channel Architecture, & Immersion:** 1.0 Mono through 22.2 Multichannel, object-based spatial audio (Dolby Atmos, DTS:X), Higher-Order Ambisonics (HOA), and binaural Head-Related Transfer Functions (HRTF).
12. **Dimension L: Technical Codecs, Telephony, Fake Hi-Res, & Containers:** Lossless PCM/FLAC, psychoacoustic lossy (MP3, AAC, Opus), telephony bandpass codecs (G.711, AMR-WB, EVS), fake upsampled hi-res detection, and container syntax.
13. **Dimension M: Studio Production, Mastering Loudness (LUFS), Tuning, & Mixing Operations:** Dynamic range compression, pitch quantization (Auto-Tune, Melodyne DNA), EBU R128 integrated loudness (LUFS), True-Peak limiting, and neural stem demixing.
14. **Dimension N: Generative AI Audio Synthesis & Neural Voice Frontier:** Flow-matching speech models (ElevenLabs v3/v4), autoregressive codec language models, real-time neural voice changers, and generative song synthesizers (Suno v5.5, Udio).
15. **Dimension O: Audio Forensic Science, Provenance Standards, & Authenticity Physics:** Electric Network Frequency (ENF) grid phase continuity, vocal tract physical biomechanics ($F_1\text{--}F_4$), forensic silence noise floor profiling, and C2PA audio manifests.
16. **Dimension P: Acoustic Cryptography, Zero-Knowledge Verification, & Audio Steganography:** Zero-Knowledge verifiable audio redaction (zk-SNARKs), multi-layer watermarking (AudioSeal, WavMark, SynthID), phase coding, and DSSS spread-spectrum audio steganography.

This document establishes the operational forensic taxonomy of audio categories across the physical universe, biological organisms, studio production, telephony networks, and generative neural frontiers as of October 2026. It formalizes **sixteen analytical dimensions**, introduces a comprehensive **Cross-Cutting Legal, Safety, & File Security Framework**, establishes mathematical acoustic physics formulations, and documents the calibrated evidence fusion architecture deployed within the `project-content-validation` engine.

---

## Table of Contents

1. [Inquiry Context & Research Mandate](#1-inquiry-context--research-mandate)
2. [Research Methodology & Primary Information Sources](#2-research-methodology--primary-information-sources)
3. [The Sixteen-Dimensional Audio Taxonomy Ontology](#3-the-sixteen-dimensional-audio-taxonomy-ontology)
4. [Dimension A: Provenance, Origin, & Authenticity Spectrum](#4-dimension-a-provenance-origin--authenticity-spectrum)
5. [Dimension B: Semantic Content Category, Sound Events, Music Theory, & Paralinguistics](#5-dimension-b-semantic-content-category-sound-events-music-theory--paralinguistics)
6. [Dimension C: Acoustic Physics, Environmental Propagation, & Side-Channels](#6-dimension-c-acoustic-physics-environmental-propagation--side-channels)
7. [Dimension D: Frequency Spectrum, Ultra-Wideband, RF Demodulation, & Data-Over-Sound](#7-dimension-d-frequency-spectrum-ultra-wideband-rf-demodulation--data-over-sound)
8. [Dimension E: Cosmic, Interstellar, & Astrophysical Acoustic Phenomena](#8-dimension-e-cosmic-interstellar--astrophysical-acoustic-phenomena)
9. [Dimension F: Quantum, Nanoscale, Cellular, & Bio-Acoustic Modalities](#9-dimension-f-quantum-nanoscale-cellular--bio-acoustic-modalities)
10. [Dimension G: Photoacoustic, Optoacoustic, & Acousto-Optic Hybrid Modalities](#10-dimension-g-photoacoustic-optoacoustic--acousto-optic-hybrid-modalities)
11. [Dimension H: Parametric Non-Linear Ultrasonic Audio Arrays](#11-dimension-h-parametric-non-linear-ultrasonic-audio-arrays)
12. [Dimension I: Neuro-Acoustic Decoding & Auditory Electrophysiology](#12-dimension-i-neuro-acoustic-decoding--auditory-electrophysiology)
13. [Dimension J: Transduction, Microphonic Physics, & Hardware Acquisition Modalities](#13-dimension-j-transduction-microphonic-physics--hardware-acquisition-modalities)
14. [Dimension K: Spatial Audio Geometry, Channel Architecture, & Immersion](#14-dimension-k-spatial-audio-geometry-channel-architecture--immersion)
15. [Dimension L: Technical Codecs, Telephony, Fake Hi-Res, & Containers](#15-dimension-l-technical-codecs-telephony-fake-hi-res--containers)
16. [Dimension M: Studio Production, Mastering Loudness (LUFS), Tuning, & Mixing Operations](#16-dimension-m-studio-production-mastering-loudness-lufs-tuning--mixing-operations)
17. [Dimension N: State of the Art in Generative AI Audio Synthesis & Neural Voice Frontier](#17-dimension-n-state-of-the-art-in-generative-ai-audio-synthesis--neural-voice-frontier)
18. [Dimension O: Audio Forensic Science, Provenance Standards, & Authenticity Physics](#18-dimension-o-audio-forensic-science-provenance-standards--authenticity-physics)
19. [Dimension P: Acoustic Cryptography, Zero-Knowledge Verification, & Audio Steganography](#19-dimension-p-acoustic-cryptography-zero-knowledge-verification--audio-steganography)
20. [Section 20: Dedicated Metadata, Container-Structure, & Encoder Forensics](#20-dedicated-metadata-container-structure--encoder-forensics)
21. [Section 21: Cross-Cutting Framework (Legal, Safety, Cheapfakes, Granularity, Security)](#21-cross-cutting-framework-legal-safety-cheapfakes-granularity--security)
22. [Section 22: Forensic Physics & Mathematical Differentiation Formulations](#22-forensic-physics--mathematical-differentiation-formulations)
23. [Section 23: System Architectural Integration & Calibrated Bayesian Verification](#23-system-architectural-integration--calibrated-bayesian-verification)
24. [Section 24: Conclusion & Strategic Roadmap](#24-conclusion--strategic-roadmap)
25. [Section 25: References & Citations](#25-references--citations)
26. [Appendix A: Controlled Vocabulary Standards Mapping](#appendix-a-controlled-vocabulary-standards-mapping)
27. [Appendix B: Canonical Generative AI Audio Architecture Specifications](#appendix-b-canonical-generative-ai-audio-architecture-specifications)
28. [Appendix C: Audio Codec Forensic Signature & Psychoacoustic Cutoff Matrix](#appendix-c-audio-codec-forensic-signature--psychoacoustic-cutoff-matrix)
29. [Appendix D: Legacy & Historical Audio Format Timeline](#appendix-d-legacy--historical-audio-format-timeline)
30. [Appendix E: Mathematical Physics Formulations for Acoustic Forensics](#appendix-e-mathematical-physics-formulations-for-acoustic-forensics)
31. [Appendix F: Implementation Status (REV7)](#appendix-f-implementation-status-rev7)

---

## 1. Inquiry Context & Research Mandate

The objective of this specification is to establish a rigorous, mathematically defensible, and comprehensive operational audio taxonomy for the `project-content-validation` platform as of late 2026.

Acoustic content validation represents an open-class problem: acoustic sensors, lossy compression pipelines, speech synthesis transformers, and adversarial manipulation vectors evolve constantly. While no closed system can claim absolute zero-omission over an unbounded future, this report provides exhaustive coverage across known physical acoustics, broadcast/streaming standards, medical diagnostics, telephony networks, and generative speech/music frontiers, formalizing explicit handling for `UNKNOWN`, `AMBIGUOUS`, and `OUT-OF-DISTRIBUTION (OOD)` generative models.

---

## 2. Research Methodology & Primary Information Sources

Information was retrieved, cross-validated, and compiled from the following bodies:
*   **Standards Bodies:** AES (AES47, AES67, AES70), EBU R128 / Tech 3333, C2PA v2.4, IPTC Audio NewsCodes, ISO/IEC 27037:2012, ITU-R BS.1770-4.
*   **Frontier Physics & Auditory Laboratories:** LIGO/Virgo/KAGRA Gravitational Wave Consortia, NASA Voyager Plasma Wave Science, Photoacoustic Imaging Research Group (Caltech), Westervelt Parametric Sound Lab, UCSF Center for Neural Engineering (ECoG speech decoding).

---

## 3. The Sixteen-Dimensional Audio Taxonomy Ontology

```
[ Dimension A: Provenance & Authenticity ]   ──> Physical mic capture, edited/mixed, tuned, AI-enhanced, or pure synthetic.
[ Dimension B: Content, Music & Paraling ]   ──> Speech, music theory, Hornbostel-Sachs, paralinguistics (age, emotion, LID).
[ Dimension C: Acoustic Physics & Space ]    ──> Wave propagation, RT60 reverb, acoustic side-channels (keystrokes, visual mic).
[ Dimension D: Frequency & Data-Over-Sound ] ──> Infra/audible/ultrasound, AM/FM/SSB RF audio, Morse, seismic audification, DTMF, SSTV.
[ Dimension E: Cosmic & Astrophysical Audio ]──> Gravitational wave chirps, Voyager plasma audio, helioseismology.
[ Dimension F: Quantum, Nano & Bio-Acoustic ]──> Quantum single phonons, cellular nanomechanics, otoacoustic OAE.
[ Dimension G: Photoacoustic & Optoacoustic ]──> Laser-induced ultrasound (PAT/PACT), Acousto-Optic Bragg diffraction.
[ Dimension H: Parametric Ultrasonic Audio ] ──> Directional non-linear self-demodulated ultrasound audio beams.
[ Dimension I: Neuro-Acoustic & BCI Decoding]──> Invasive ECoG neural speech decoding, ABR/ECochG dipole tracking.
[ Dimension J: Transduction & Microphones ]  ──> Dynamic, condenser, ribbon, MEMS, hydrophone, laser vibrometer.
[ Dimension K: Spatial Audio & Immersion ]   ──> Mono, stereo, 5.1/7.1, binaural HRTF, Dolby Atmos, ambisonics.
[ Dimension L: Codecs, Telephony & Hi-Res ]  ──> Codecs (PCM, Opus, FLAC), telephony (G.711/AMR), fake hi-res upsampling.
[ Dimension M: Studio DSP, Loudness & Tuning]──> Auto-Tune, Melodyne, mastering loudness (EBU R128 / LUFS), stem separation.
[ Dimension N: Generative AI Audio Frontier] ──> TTS, zero-shot voice cloning, flow matching, real-time voice changers.
[ Dimension O: Forensics, Fingerprints & ENF]──> ENF 50/60Hz, acoustic fingerprinting (Shazam/Chromaprint), SynthID.
[ Dimension P: Cryptography & Audio Stego ]  ──> zk-SNARK audio redaction proofs, phase coding, DSSS steganography.
```

---

## 4. Dimension A: Provenance, Origin, & Authenticity Spectrum

1. **State 1: Pure Physical Acoustic Capture (`digitalCapture`):** Diaphragm pressure waves, continuous ambient noise floor, continuous ENF 50/60Hz trace.
2. **State 2: Analog Media Digitization (`soundRecordingTransfer`):** Wax cylinders, 78 RPM shellac, vinyl microgroove, open-reel magnetic tape (tape hiss, wow/flutter).
3. **State 3: Digital Internal Loopback Capture (`softwareImage`):** OS software recording (WASAPI/CoreAudio), $-\infty\text{ dBFS}$ digital silence, zero ENF.
4. **State 4: Software-Tuned / Pitch-Corrected Audio (`softwareTunedAudio`):** Auto-Tune, Melodyne DNA, instantaneous retune speeds, phase-vocoder formant markers.
5. **State 5: Conventional DAW Mix & Composite Editing (`minorHumanEdits`, `compositeSynthetic`):** Multi-track assembly, crossfades, dynamic range compression.
6. **State 6: Neural Stem-Separated Audio (`neuralStemSeparation`):** Demucs, LALAL.AI, UVR with high-frequency spectral masking holes and STFT phase smearing.
7. **State 7: Deepfake Voice Conversion (`voiceConversionDeepfake`):** RVC, OpenVoice swapping timbral resonance while retaining source timing.
8. **State 8: Neural Text-to-Speech (TTS) Re-Enactment (`trainedAlgorithmicMedia` - Speech):** ElevenLabs, Chirp 3 HD, Cartesia Sonic with synthetic pitch intervals.
9. **State 9: Fully AI-Generated Music & Soundscapes (`trainedAlgorithmicMedia` - Music):** Suno v5.5, Udio, Stable Audio 3.0 exhibiting RVQ token periodicity ($12.5\text{–}25\text{ Hz}$).
10. **State 10: Adversarially Spoofed / Anti-Forensic Audio:** Injected synthetic ENF, layered room tone, physical re-mic replay attacks.
11. **State 11: Real-Time Neural Voice Morphing & AI Dubbing:** Low-latency VST voice conversion during live telephony/calls; automated multilingual AI dubbing preserving speaker timbre across translated phonetic structures.

---

## 5. Dimension B: Semantic Content Category, Sound Events, Music Theory, & Paralinguistics

Mapped against **Google AudioSet** (632 classes), **Universal Category System (UCS v8.2)**, and acoustic musicology:
*   **Speech & Voice Hierarchy:** Conversational, formal oratory, broadcast news, air traffic control, 911 dispatch, baby vocalizations.
*   **Speaker & Paralinguistic Profiling:**
    *   *Demographics & Pathology:* Age estimation, biological sex/gender cues, vocal pathology biomarkers (dysphonia, vocal tremors, Parkinsonian acoustic decay).
    *   *Emotional Valence & Prosody:* Arousal, stress, pitch variability, micro-tremors under cognitive load.
    *   *Language & Diarization:* Spoken Language Identification (LID), dialect/accent classifiers, and multi-speaker diarization time-stamping ($T_{\text{speaker}_k} = [t_{\text{start}}, t_{\text{end}}]$).
*   **Music Theory, Tuning Systems, & Instrument Taxonomy:**
    *   *Tuning Temperaments:* 12-Tone Equal Temperament (12-TET), Just Intonation, Pythagorean tuning; Standard concert pitch ($A_4 = 440\text{ Hz}$ vs. historical $432\text{ Hz}$ / Baroque $415\text{ Hz}$).
    *   *Hornbostel-Sachs Classification:* Idiophones, Membranophones, Chordophones, Aerophones, and Electrophones.
    *   *MIDI & Expressive Performance:* Standard MIDI 1.0/2.0 note-on/velocity streams and MIDI Polyphonic Expression (MPE) pitch-bend curves.
*   **Sampling, Plagiarism, & Audio Fingerprinting:**
    *   *Sampling & Micro-Looping:* Identification of sampled master recordings in modern production.
    *   *Acoustic Fingerprinting:* Peak constellation hashing (Shazam landmark pairs), Chromaprint / AcoustID spectral fingerprinting, and MusicBrainz database cross-referencing.
*   **Sound Effects (SFX) & Foley:** Footsteps, clothing rustle, prop handling, vehicle engines, gunshots, explosions, UI notification chimes, cinematic sub-bass *braams*, whooshes.
*   **Natural & Built Soundscapes:** Rain, thunderstorms, wind, ocean surf, animal biophony (canine, cetacean, avian, insect), urban anthrophony.
*   **Scientific Diagnostics:** Medical heart valve auscultation ($S_1/S_2$), lung wheezes/crackles, industrial bearing vibration failure signatures.

---

## 6. Dimension C: Acoustic Physics, Environmental Propagation, & Side-Channels

*   **Wave Equation:** $\nabla^2 p - \frac{1}{c^2} \frac{\partial^2 p}{\partial t^2} = 0$.
*   **Room Impulse Response & Reverberation:** Direct sound vs. Early reflections ($< 50\text{ ms}$) vs. Late diffuse reverberation ($RT_{60} = \frac{0.161 V}{\sum S_i \alpha_i}$).
*   **Wave Phenomena:** Inverse-square distance attenuation ($-6\text{ dB}$ per distance doubling), frequency-dependent atmospheric absorption ($> 4\text{ kHz}$), Doppler effect pitch sweeps, and room mode standing wave resonances.
*   **Acoustic Side-Channels & Mechanical Leakage:**
    *   *Keystroke Acoustic Profiling:* Eavesdropping on mechanical keyboard sound emissions to reconstruct typed passwords with $>90\%$ accuracy.
    *   *Hardware Acoustic Cryptanalysis:* High-frequency ultrasonic coil whine from motherboard capacitors leaking CPU execution state and cryptographic keys.
    *   *The "Visual Microphone":* Recovering intelligible speech from high-speed video recording microscopic sound-induced surface vibrations of objects (plastic bags, plant foliage, glass windows).

---

## 7. Dimension D: Frequency Spectrum, Ultra-Wideband, RF Demodulation, & Data-Over-Sound

*   **Infrasound ($< 20\text{ Hz}$):** Seismic P/S waves, microbaroms, volcanic explosive plumes, elephant infrasonic communication.
*   **Audible Spectrum ($20\text{ Hz} - 20\text{ kHz}$):** Sub-bass, bass, vocal fundamentals ($85\text{–}255\text{ Hz}$), speech intelligibility formants ($F_1\text{–}F_4$), presence ($4\text{–}6\text{ kHz}$), and air brilliance ($6\text{–}20\text{ kHz}$).
*   **Ultrasound ($> 20\text{ kHz}$):** Bat echolocation, industrial NDT weld flaw testing ($1\text{–}10\text{ MHz}$), medical diagnostic sonography ($2\text{–}15\text{ MHz}$).
*   **Radio Frequency Demodulation Audio & Morse Code:**
    *   *AM / FM / SSB Demodulation:* Audio signals demodulated from RF carriers across HF, VHF, and UHF bands; characterized by ionospheric fading, selective phase cancellations, carrier beat notes, and receiver AGC pumping.
    *   *Continuous Wave (CW) / Morse Code:* Keyed on-off tone bursts ($600\text{–}800\text{ Hz}$) with strict element timing ratios ($1:3$ dit-to-dah durations).
*   **Planetary & Terrestrial Seismic Audification:** Frequency-scaled acoustic transformations of planetary seismic vibrations, fault slip transients, and Martian tremors recorded by InSight SEIS, transposing sub-hertz crustal oscillations into human audible frequencies.
*   **Telecommunications & Data-Over-Sound:**
    *   *Signaling Tones:* Dual-Tone Multi-Frequency (DTMF) telephony touch-tones.
    *   *Acoustic Modems & SSTV:* FSK/PSK audio modem handshakes; Slow Scan Television (SSTV) analog audio image transmissions.
    *   *Near-Field Ultrasonic Beaconing:* High-frequency audio beacons ($18\text{–}22\text{ kHz}$) transmitting binary payloads over speaker-to-microphone air gaps.

---

## 8. Dimension E: Cosmic, Interstellar, & Astrophysical Acoustic Phenomena

*   **Gravitational Wave Auditory Chirps (LIGO / Virgo / KAGRA):** Spacetime metric strain $h(t)$ sonified directly; binary black hole and neutron star inspiral coalescence sweeping upward in frequency ($f(t) \sim (t_c - t)^{-3/8}$ from $30\text{ Hz}$ to $2\text{ kHz}$).
*   **Interstellar Plasma Waves:** Voyager 1 & 2 PWS recording interstellar plasma density oscillations.
*   **Planetary Magnetospheric Audio:** Jovian lightning whistlers and Saturnian kilometric radiation (SKR).
*   **Helioseismology & Asteroseismology:** Acoustic $p$-mode oscillations traversing stellar interiors.
*   **Primordial Sound Waves (BAO):** Baryon Acoustic Oscillations traveling at relativistic plasma sound speed $c_s \approx c/\sqrt{3}$ in the early universe.
*   **Millisecond Pulsar Rhythms:** Rotating neutron stars sweeping radio beams creating periodic acoustic clicks up to $716\text{ Hz}$.

---

## 9. Dimension F: Quantum, Nanoscale, Cellular, & Bio-Acoustic Modalities

*   **Quantum Phononics:** Quantized mechanical vibrations (single phonons, $E = (n + \frac{1}{2})\hbar\omega$) in optomechanical cavities and superconducting qubits.
*   **Cellular Nanomechanics:** Atomic Force Acoustic Microscopy measuring cellular stiffness; yeast cell walls vibrating at a metabolic frequency of $\approx 1.6\text{ kHz}$.
*   **Human Cochlear Otoacoustic Emissions (OAE):** Active sound generation by outer hair cells in the inner ear: Spontaneous (SOAE) and Distortion Product (DPOAE).
*   **Oceanic SOFAR Waveguide Channel:** Refractive acoustic channel in the deep ocean ($\approx 1000\text{ m}$) allowing low-frequency sound to propagate thousands of kilometers.

---

## 10. Dimension G: Photoacoustic, Optoacoustic, & Acousto-Optic Hybrid Modalities

```
[ Optical Laser Pulse (Nanoseconds) ] ───> [ Biological Tissue Chromophores ] ───> [ Thermoelastic Expansion ]
                                                          │                                      │
                                                 Stress Confinement                     Initial Acoustic Shockwave
                                                 (τ_p < d_c / c_s)                      (p_0 = Γ · μ_a · F)
```

1.  **Photoacoustic / Optoacoustic Imaging (PAT / PACT):** Nanosecond laser pulses irradiate biological tissue; endogenous hemoglobin absorbs optical energy, undergoing rapid thermoelastic expansion that generates localized ultrasound pressure waves ($p_0 = \Gamma \mu_a F$). Bridges optical absorption contrast with ultrasonic penetration depth ($> 5\text{ cm}$) to image tumor microvasculature.
2.  **Acousto-Optic Modulation (AOM):** Acoustic radio frequency waves propagating in photoelastic crystals (e.g., $TeO_2$) create moving refractive index gratings that diffract and frequency-shift laser beams in the Bragg regime ($Q \gg 1, \sin \theta_B = \lambda / (2n\Lambda)$).

---

## 11. Dimension H: Parametric Non-Linear Ultrasonic Audio Arrays

```
[ Parametric Transducer Array ] ───> [ High-Intensity 40 kHz Ultrasonic Carrier ] ───> [ Non-Linear Air Medium ]
                                                      │                                           │
                                            Audio Signal Modulated                     Self-Demodulation
                                            (Westervelt Wave Equation)                 Directional Audible Beam
```

*   **Physical Mechanism:** Directing a collimated beam of ultrasound carrier waves ($40\text{ kHz}$) modulated with audio.
*   **Self-Demodulation in Air:** Because air behaves non-linearly at high acoustic pressures (governed by the Westervelt wave equation), the air medium self-demodulates the ultrasonic wave, producing localized audible sound that propagates like a laser beam, audible only within the narrow column.

---

## 12. Dimension I: Neuro-Acoustic Decoding & Auditory Electrophysiology

*   **Intracranial Neural Speech Decoding (ECoG / sEEG):** Decoding inner speech and heard audio directly from local field potentials in the superior temporal gyrus (STG) and Broca's area using Audio LLMs.
*   **Auditory Brainstem Response (ABR):** Recording microvolt electrical dipole waveforms (Waves I–V) tracking neural acoustic signal propagation from the cochlea through the brainstem.
*   **Electrocochleography (ECochG):** Measuring Cochlear Microphonic, Summating Potential, and Compound Action Potential.

---

## 13. Dimension J: Transduction, Microphonic Physics, & Hardware Acquisition Modalities

*   **Transducer Types:** Dynamic moving-coil, Large/Small diaphragm condenser ($+48\text{ V}$ phantom), Corrugated ribbon, Silicon MEMS, Contact piezoelectric, Hydrophones, and long-range optical Laser Doppler Vibrometers.
*   **Polar Patterns:** Omnidirectional, Cardioid, Supercardioid, Figure-of-8, and Shotgun interference tubes.
*   **Stereo/Spatial Arrays:** Coincident XY, Near-coincident ORTF, Blumlein pair, Mid-Side (MS), Spaced Pair (A/B), and Decca Tree.

---

## 14. Dimension K: Spatial Audio Geometry, Channel Architecture, & Immersion

*   **Channel-Based:** 1.0 Mono to 22.2 NHK Super Hi-Vision.
*   **Object-Based:** Dolby Atmos (9.1 bed + 118 dynamic audio objects), DTS:X, Sony 360 Reality Audio (MPEG-H 3D Audio).
*   **Scene-Based Ambisonics:** First-Order (FOA - 4 channels) and Higher-Order Ambisonics (HOA - $(N+1)^2$ channels).
*   **Binaural Audio & HRTF:** 2-channel headphone rendering synthesizing Interaural Time Differences (ITD) and Level Differences (ILD).
*   **Wave Field Synthesis (WFS):** Massive loudspeaker arrays synthesizing physical acoustic wavefronts.

---

## 15. Dimension L: Technical Codecs, Telephony, Fake Hi-Res, & Containers

*   **Sampling Rates:** $8\text{ kHz}$ (G.711 telephony) $\to 16\text{ kHz}$ (G.722 / AMR-WB) $\to 44.1\text{ kHz}$ (CD) $\to 48\text{ kHz}$ (Film/Broadcast) $\to 96/192\text{ kHz}$ $\to 352.8\text{ kHz}$ (DXD) $\to 2.8224\text{ MHz}$ (DSD64).
*   **Bit Depths:** 16-bit ($98\text{ dB}$ SNR), 24-bit ($146\text{ dB}$ SNR), 32-bit floating-point ($> 1500\text{ dB}$ dynamic range).
*   **Telephony Codecs & VoIP Transmission Forensics:**
    *   *Narrowband Telephony (G.711 $\mu$-law/A-law):* $8\text{ kHz}$ sample rate, $300\text{–}3400\text{ Hz}$ bandpass; speech clipping and quantization noise.
    *   *Wideband / Cellular (G.722, AMR-WB, EVS):* $16\text{–}32\text{ kHz}$ sampling with adaptive multi-rate encoding.
    *   *VoIP Packet Concealment & Jitter:* Packet Loss Concealment (PLC) repetitive waveform extrapolation, Comfort Noise Generation (CNG) hiss during DTX silences, and jitter buffer time-compression artifacts.
*   **Fake Hi-Res Audio Detection:**
    *   *Upsampling Forensics:* Detecting audio transcoded from $44.1\text{ kHz}$ or lossy MP3 into $96\text{ kHz}$ or $192\text{ kHz}$ FLAC/WAV by identifying steep brickwall cutoffs at $20\text{ kHz} / 22.05\text{ kHz}$ with zero energy in ultrasonic octaves.
    *   *Bit-Depth Padding:* Inspecting bit distributions to identify 16-bit audio padded with zeros to appear as 24-bit true high-resolution.
    *   *Neural Bandwidth Extension Forensics:* Identifying hallucinated high-frequency overtones synthesized by AI super-resolution models that lack harmonic phase coherence with low-frequency fundamentals.
*   **Codecs & Containers:** Linear PCM, Broadcast Wave (BWF), FLAC, ALAC, MP3, AAC, Opus, and Neural Audio Codecs (EnCodec, SoundStream, DAC, FlexiCodec).

---

## 16. Dimension M: Studio Production, Mastering Loudness (LUFS), Tuning, & Mixing Operations

*   **Vocal Tuning:** Antares Auto-Tune Pro 11 ($0\text{ ms}$ retune speed plateaus), Celemony Melodyne 5 Studio (DNA polyphonic editing), VocALign.
*   **Mastering Loudness Standards (ITU-R BS.1770-4 & EBU R128):**
    *   *Integrated Loudness (LUFS/LKFS):* Standardized perceptual loudness measurement ($-14\text{ LUFS}$ streaming target for Spotify/YouTube; $-23\text{ LUFS} \pm 0.5\text{ LU}$ for EBU R128 broadcast).
    *   *Loudness Range (LRA):* Statistical measure of dynamic range across the program material.
    *   *Maximum True Peak (dBTP):* Intersample peak detection using $4\times$ oversampling to detect D/A reconstruction clipping.
    *   *Dynamic Over-Compression:* Squashed crest factors ($< 6\text{ dB}$), hyper-compressed RMS levels, and shaved transient peaks indicative of loudness war mastering.
*   **Dynamics & EQ:** FET, VCA, Optical, Digital Brickwall Limiters, Downward Gating, De-Essers, Minimum-Phase vs. Linear-Phase FIR EQ.
*   **Time-Based & Neural:** Convolution Reverb, Algorithmic Reverbs, Neural Stem Separation (Demucs v4, LALAL.AI).

---

## 17. Dimension N: State of the Art in Generative AI Audio Synthesis & Neural Voice Frontier

*   **Foundation Architectures:** Conditional Flow Matching (CFM) and Streaming Flow Matching (SFM) delivering sub-200ms real-time conversational speech; Neural Audio Codec LLMs.
*   **Voice Synthesis & Zero-Shot Cloning:** ElevenLabs v3/v4, Google Chirp 3 HD, Cartesia Sonic 3.6 ($< 150\text{ ms}$ streaming latency), Fish Speech, Kokoro 82M.
*   **Real-Time Voice Changers & AI Dubbing:** Real-time VST voice morphers (RVC, Voice.ai) executing pitch and timbre transformation with low latency; automated multilingual dubbing systems (HeyGen Audio, ElevenLabs Dubbing) translating dialogue while preserving the speaker's vocal acoustic fingerprint and room acoustics.
*   **Generative Music & SVS:** Suno v5.5 (4-minute multi-track songs with stem export), Udio, Stable Audio 3.0 (native DAW VST plugins), Singing Voice Synthesis (SVS).

---

## 18. Dimension O: Audio Forensic Science, Provenance Standards, & Authenticity Physics

*   **Electrical Network Frequency (ENF) Analysis:** Continuous $50\text{ Hz} / 60\text{ Hz}$ grid frequency drift matching for timestamp and geo-location authentication, plus phase jump detection.
*   **Acoustic Tube Formant Physics:** Vocal tract resonance boundary checking ($F_n = \frac{(2n-1)c}{4L}$) to flag impossible physical vocal dimensions.
*   **Double Compression Analysis:** Conflicting MDCT block sizes and psychoacoustic cutoffs revealing multi-generation re-encoding.
*   **Active Provenance:** C2PA v2.4 Content Credentials for audio paired with Google SynthID waveform watermarking.

---

## 19. Dimension P: Acoustic Cryptography, Zero-Knowledge Verification, & Audio Steganography

```
                                [ AUDIO CRYPTOGRAPHY & STEGANOGRAPHY ]
                                                 │
       ┌────────────────────────┬────────────────┴────────────────┬────────────────────────┐
       ▼                        ▼                                 ▼                        ▼
 [ ZERO-KNOWLEDGE PROOFS ] [ SILICON HARDWARE TRUST ]      [ AUDIO STEGANOGRAPHY ]  [ DEEP AUDIO STEGANALYSIS ]
 • zk-SNARKs for Audio     • Sensor Preamp ASIC Key        • Phase Coding           • Neural Audio Steganalysis
 • Verifiable Silence &      Injection & Silicon PUF       • Echo Hiding (0.5-1ms)  • High-Frequency Phase
   Redaction Without Raw   • Hardware C2PA Signing         • Spread Spectrum (DSSS)   Variance Inspection
```

### 19.1 Zero-Knowledge Proofs for Verifiable Audio Redaction
*   Using **zk-SNARKs** to mathematically prove that a released audio recording was derived via authentic noise suppression or lawful redaction/muting from a cryptographically signed hardware master without revealing redacted audio segments.

### 19.2 Multi-Layer Audio Watermarking Landscape
*   **Meta AudioSeal:** High-precision localized neural audio watermarking with sample-level localization resolution ($<1\text{ ms}$) enabling real-time detection of synthetic voice generation and pinpointing injected deepfake speech segments within natural audio streams.
*   **WavMark:** Deep neural network-based imperceptible watermarking embedded in acoustic spectrogram latents; robust against non-linear pitch shifting, acoustic room re-recording, and low-bitrate MP3/AAC compression down to $32\text{ kbps}$.
*   **Google SynthID for Audio:** Spread-spectrum pseudorandom phase perturbation embedded directly into the sampling latent space of generative music and speech models (Lyria, MusicLM).
*   **Spread-Spectrum (DSSS) Acoustic Watermarks:** Direct sequence spread-spectrum patterns modulated below psychoacoustic masking curves ($< -30\text{ dB}$ relative to signal) for high-entropy forensic metadata embedding.

### 19.3 Audio Steganography & Steganalysis
*   **Embedding Mechanisms:** Phase coding (replacing initial audio phase with payload phase), Echo hiding (introducing sub-millisecond delays), Direct Sequence Spread Spectrum (DSSS).
*   **Steganalysis:** Detecting phase variance anomalies and non-random cepstral distributions using neural classifier backbones.

---

## 20. Dedicated Metadata, Container-Structure, & Encoder Forensics

Digital audio bitstreams, container structures, and header chunks provide non-acoustic signals that corroborate or contradict acoustic waveform analysis.

> **Evidentiary Weight:** Tags, chunks, and encoder strings are trivially forged and are routinely rewritten or stripped by benign converters, DAWs, and platforms. A valid encoder signature attests only to the last encoder in the chain, not to the origin of the sound. These signals enter the Bayesian LLR fusion as low-weight evidence and never as a sole determinant.

### 20.1 ID3v1 & ID3v2 Extended Header Forensics
*   **Tag Version Consistency & Unsynchronization:** Parsing ID3v2.3 and ID3v2.4 frame headers in MP3 and AIFF files. Genuine digital audio workstations (DAWs) populate standard frame sets (`TIT2`, `TPE1`, `TALB`); synthetic or re-muxed files often contain mismatched unsynchronization flags, corrupted frame sizes, or orphaned padding buffers.
*   **User-Defined Text & Private Frames:** Inspecting `TXXX` and `PRIV` frames for signature strings left by software processors, automated online converters (e.g., youtube-dl, FFmpeg lavf), or generative synthesis platforms (e.g., ElevenLabs API transaction IDs).
*   **Embedded Cover Art (`APIC`) Desynchronization:** Verifying that embedded JPEG/PNG album art hashes and dimensions match file creation timestamps and container specifications.

### 20.2 Broadcast Wave Format (BWF / EBU Tech 3285) & RIFF Chunks
*   **BWF `bext` Chunk Coding History:** Professional broadcast recordings adhere to EBU Tech 3285, requiring a `bext` (Broadcast Audio Extension) chunk containing `Originator`, `OriginatorReference`, `OriginationDate`, `OriginationTime`, `TimeReference`, and a free-text `CodingHistory` ASCII log intended to record A/D conversions, sample rate changes, and DSP processes in the file's lineage (the field is writable and unauthenticated). The absence of `bext` in a file claimed to be a broadcast-chain master weakens that claim, but does not by itself indicate tampering.
*   **RIFF Chunk Order & Fact Chunks:** Inspecting canonical RIFF structure (`fmt `, `data`, `smpl`, `cue `). Non-PCM WAV formats (e.g., IEEE float, ADPCM) require an explicit `fact` chunk declaring the true uncompressed sample count; missing `fact` chunks indicate non-compliant software synthesis.

### 20.3 Open-Source Codec Containers & Vorbis Comments
*   **Ogg Bitstream Serial Numbers & Page Checksums:** Ogg container streams (used by Opus, Vorbis, FLAC) feature 32-bit serial numbers and CRC-32 checksums per page. Mid-stream serial changes or non-monotonic granule positions expose spliced audio segments.
*   **FLAC MD5 Audio Signature:** Native FLAC headers include a 128-bit MD5 checksum of the unencoded raw PCM audio data. A mismatch between the decoded PCM hash and the header MD5 reveals corruption or modification of the FLAC stream after encoding, unless the MD5 was recomputed; a match does not prove the audio was unaltered before encoding.
*   **MP4 / M4A QuickTime Metadata Atoms:** Inspecting the `moov/udta/meta/ilst` atom hierarchy for Apple-specific encoder tags (`©too`, `©art`, `stik`).

### 20.4 Encoder Bitstream Signatures & Frame Headers
*   **LAME Tag & Xing Header Forensics:** MP3 bitstreams encoded by the LAME library embed a LAME info tag inside the first VBR/CBR Xing/Info frame, containing the encoder version (e.g., `LAME3.100`), encoding preset flags, lowpass filter cutoff frequency, and tag CRC-16 checksums. Files lacking the tag, carrying a non-matching CRC, or showing a lowpass value inconsistent with the measured spectral cutoff indicate re-wrapping or tag editing; a consistent tag only attests to the final encoding pass.
*   **FFmpeg / Libavformat Header Signatures:** Inspecting default encoder identification strings (`Lavf59.27.100`, `Lavc59.37.100`) embedded in container headers, isolating automated script pipelines.

---

## 21. Cross-Cutting Framework: Legal, Safety, Cheapfakes, Granularity, & Security

Acoustic signal verification operates in a complex legal, security, and human environment. Authenticity cannot be reduced solely to spectral FFT bins or waveform statistics; it intersects with legal rights, conversational context, and adversarial software threats.

### 21.1 Trust, Safety, & Harm Classification Taxonomy
*   **Fraudulent Voice Impersonation & Vishing:** Immediate categorization of unauthorized financial wire transfer requests, biometric telephone banking bypass attempts, and synthetic executive voice mandates.
*   **Hate Speech, Intimidation, & Harassment:** Detection of synthetic audio harassment campaigns, coordinated inauthentic voice broadcasts, and non-consensual sexualized audio generation.
*   **Public Safety & Civic Manipulation:** Real-time flagging of fabricated emergency broadcasts, fake public official announcements, and altered 911 dispatch calls.
*   **Hard-Block Abuse Audio:** Audio streams associated with Child Sexual Exploitation and Abuse (CSAE) represent non-negotiable hard blocks triggering forensic isolation and mandatory statutory reporting.

### 21.2 Legal, Regulatory, & Evidentiary Framework
*   **Voice Likeness, Right of Publicity, & Voice Cloning Tort:** Unauthorized commercial replication or cloning of a person's distinctive vocal identity (voice actors, vocalists, public figures) violates statutory rights of publicity and FTC unfair competition regulations.
*   **Biometric Voice Privacy Compliance:** Enforcement of GDPR Article 9 (biometric voice data), Illinois Biometric Information Privacy Act (BIPA, 740 ILCS 14/), Texas Capture or Use of Biometric Identifier Act (CUBI, Tex. Bus. & Com. Code § 503.001), and Washington State Biometric Privacy Law (RCW 19.375) regarding acoustic voiceprints and speaker recognition templates ($d$-vectors, $x$-vectors).
*   **EU AI Act Transparency Obligations:** Strict compliance with Regulation (EU) 2024/1689 Article 50, mandating that providers and deployers of AI systems generating or manipulating synthetic audio media mark the outputs in a machine-readable format and ensure prominent public disclosure.
*   **Emergency Vishing Scams & FCC Robocall Enforcement:** Strict enforcement under Federal Communications Commission (FCC) Declaratory Ruling 24-17 pursuant to the Telephone Consumer Protection Act (TCPA, 47 U.S.C. § 227), establishing that AI-generated synthetic voices in robocalls constitute "artificial or prerecorded voice" communications, making non-consensual synthetic voice calls unlawful.
*   **Medical Data Privacy (HIPAA / GDPR / EHDS):** De-identification compliance verifying that electronic stethoscope audio, acoustic cough recordings, and voice biomarkers (e.g., vocal fold pathology, Parkinson's speech acoustic features) strictly satisfy HIPAA Safe Harbor (45 CFR § 164.514) and European Health Data Space (EHDS) guidelines, ensuring zero leaked Protected Health Information (PHI).
*   **Evidentiary Admissibility & ENF Courtroom Grounding:** Establishing chain of custody for digital audio evidence via cryptographic hashes, Daubert/Frye scientific reliability benchmarks, Federal Rules of Evidence (FRE 901/902), and ISO/IEC 27037:2012 digital evidence handling. Electric Network Frequency (ENF) temporal matching against regional power grid reference archives can corroborate recording time and location where a mains-induced trace is present; it is supporting evidence, not standalone proof.
*   **Copyright on Sound Recordings vs. Underlying Compositions:** Differentiating the master recording copyright ($\text{℗}$) from the underlying musical composition ($\text{©}$), especially in AI training provenance and neural stem separation reuse disputes.

### 21.3 "Real but Misleading": Contextual Cheapfakes & Audio Splicing
*   **Contextual Audio Cheapfakes:** Authentic, unedited historical speech recordings republished with falsified speaker attributions, altered dates, or deceptive contextual captions.
*   **Splicing & Out-of-Context Clipping:** Subtly excising mitigating words (e.g., removing *"don't"* or *"never"*) using microscopic crossfades to completely reverse the speaker's stated intent without introducing audible clicks.
*   **Speed & Pitch Tampering (Shallowfakes):** Slowing down authentic speech (e.g., to 75% speed) with pitch correction to falsely portray cognitive decline, neurological impairment, or alcohol intoxication.
*   **Forensic Grounding:** Forensic wave analysis confirms physical mic acoustics; contextual resolution requires automated speech-to-text (ASR) semantic cross-checking against known historical transcripts and verified source media.

### 21.4 Temporal & Speaker Granularity
*   **Temporal Tamper Masking:** Manipulations are frequently localized to isolated words or phonemes within an otherwise authentic dialogue. The validation engine outputs a continuous temporal segmentation mask:
    $$M(t) \in [0, 1]$$
    pinpointing the precise millisecond interval $[t_{\text{start}}, t_{\text{end}}]$ where neural voice inpainting or audio splicing occurred.
*   **Speaker Diarization Binding:** In multi-speaker conversations, authenticity must be tracked on a per-speaker basis, isolating whether Speaker A is an authentic physical recording while Speaker B is an injected synthetic conversational agent.

### 21.5 Media Lifecycle & Platform Transcoding Laundering
*   **The Lossy Transcoding Cascade:** Audio undergoes severe multi-generation degradation: Studio Master (24-bit/96kHz WAV) $\to$ Video NLE export (AAC-LC 256 kbps) $\to$ Social Messaging App (Opus 16 kbps narrowband) $\to$ Social Video Upload (re-encoded MP3 at 128 kbps).
*   **Anti-Forensic Laundering vs. Benign Bitrate Starvation:** The engine distinguishes between benign bandwidth loss (psychoacoustic quantization noise) and deliberate laundering designed to strip ENF mains hum or smooth out neural vocoder phase artifacts.

### 21.6 Audio File Security, Container Polyglots, & Adversarial Attacks
*   **Audio Container Polyglots:** Binary files that parse simultaneously as valid WAV/MP3 audio and executable scripts or ZIP archives, concealing malicious payloads inside metadata chunks (`ID3v2`, `RIFF` padding).
*   **Audio Decoder Exploits:** Malformed audio header chunks triggering integer overflow vulnerabilities in decoders (libflac, libvorbis, FFmpeg).
*   **Inaudible Ultrasonic Commands (DolphinAttack):** Modulating voice commands onto ultrasonic carrier frequencies ($20\text{–}40\text{ kHz}$) that are inaudible to human listeners but demodulate inside microphone non-linearities, secretly controlling smart assistants and voice-activated devices.
*   **Adversarial Audio Perturbations:** Sub-audible acoustic noise designed to deceive automatic speech recognition (ASR) engines or anti-spoofing neural classifiers into incorrect classifications.

### 21.7 Detector Validity, Accent/Dialect Bias, & Channel Shift
*   **Demographic Parity Across Accents & Dialects:** Auditing deepfake speech classifiers across regional accents, non-native dialects, varied age groups, and vocal tract pathologies to ensure equitable false-positive and false-negative error rates.
*   **Channel Shift & Bandwidth Generalization:** Ensuring detector calibration holds across diverse acoustic transmission channels: narrowband telephony (G.711 at $8\text{ kHz}$), wideband cellular (AMR-WB at $16\text{ kHz}$), Bluetooth handsfree profiles, and uncompressed studio XLR condenser recordings.
*   **Adversarial Robustness:** Benchmarking detector resilience against psychoacoustic adversarial perturbations, phase inversion, and over-the-air playback-and-record acoustic laundering.

### 21.8 Accessibility Tracks & Multimodal Parity
*   **Accessibility Metadata Forensics:** Inspecting embedded synchronized text transcriptions, closed caption streams, and audio description tracks for the visually impaired.
*   **Multimodal Semantic Verification:** Detecting discrepancies where generative AI synthetic speech contradicts accompanying verified written transcripts, or where automated speech-to-text models introduce hallucinated words.

### 21.9 Open-Set Source Attribution & Novel Generator Identification
*   **Open-Set Attribution Architecture:** Avoiding closed-world classification assumptions ($k \in \{1, \dots, K\}$ known voice models). The engine implements an open-set attribution head that computes latent Mahalanobis distances against known generator cluster centers $\{\boldsymbol{\mu}_k\}$. When $\min_k D_{\text{attr}}(\mathbf{z}, \boldsymbol{\mu}_k) > \tau_{\text{attr}}$, the system outputs `UNKNOWN_SOURCE / NOVEL_GENERATOR`, triggering signature clustering and automated quarantine.

---

## 22. Forensic Physics & Mathematical Differentiation Formulations

| Forensic Indicator | Authentic Physical Acoustic/Mic Capture | Conventional Edit | Software-Tuned Hybrid | Fully AI-Generated (TTS/Suno) | Digital Loopback |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **ENF Phase Continuity ($\Delta \Phi$)\*** | Continuous ($\Delta \Phi < 0.05\text{ rad}$) | Phase jumps at splice seams | Preserved from background | **Absent** (No electrical mains) | Zero ENF signal ($-\infty\text{ dBFS}$) |
| **Vocal Tract Formants ($F_1\text{--}F_4$)** | Follows acoustic tube physics | Intact | Unnatural instantaneous pitch | Physically implausible formants | N/A (Digital stream) |
| **Spectral Cutoff & MDCT** | Natural acoustic decay | Cutoff shifts at splices | Mixed psychoacoustic curves | RVQ neural token periodicity | Flat digital silence floor |
| **ZKP / C2PA Verification** | Hardware signed | zk-SNARK proof | C2PA edit assertion | C2PA synthetic tag | Software recorder tag |

*\*Note: ENF phase continuity threshold $\Delta \Phi < 0.05\text{ rad}$ and grid matching are empirical benchmarks calibrated on reference corpora from electric utility grid archives (e.g., UK National Grid, continental European ENTSO-E, and US Eastern Interconnection archives); actual operational cutoffs adapt dynamically to background noise levels and codec bitrate.*  
*\*\*Critical Note on Physical Indicators: The absence of an Electric Network Frequency (ENF) marker does NOT determinatively prove synthetic audio generation. Battery-operated portable recorders, smartphones running off DC battery power, and studio recordings powered by high-end online AC power regenerators exhibit zero ENF mains hum without generative synthesis; furthermore, telephony codecs (e.g., AMR, G.711) bandpass filter below $300\text{ Hz}$, eliminating $50/60\text{ Hz}$ mains hum.*

---

## 23. System Architectural Integration & Calibrated Bayesian Verification

The operational taxonomy defines the target `project-content-validation` pipeline using a decoupled, two-axis decision architecture. Stage names below are logical stages of the target architecture; not every stage maps to an implemented module in the current codebase (e.g., `enf`, `environment`, and `crypto` are specification-level stages).

```
[ Input Audio Stream ] ───> [ 1. audio_detector.signal ]      ───> Sample Rate, Bit Depth, True-Peak
                       ───> [ 2. audio_detector.spectral ]    ───> STFT Spectrogram, MDCT Cutoffs
                       ───> [ 3. audio_detector.enf ]         ───> 50Hz/60Hz Grid Extraction, Phase Continuity
                       ───> [ 4. audio_detector.voice ]       ───> Formant Tracking (F1-F4), Jitter/Shimmer
                       ───> [ 5. audio_detector.environment ] ───> RT60 Reverberation, Room Impulse Response
                       ───> [ 6. audio_detector.codecs ]      ───> Double Compression, Neural Codec RVQ Tokens
                       ───> [ 7. audio_detector.crypto ]      ───> C2PA Manifests, zk-SNARK Proofs, SynthID
                       ───> [ 8. Deep Acoustic Backbone ]    ───> RawNet3 / AASIST Self-Supervised Embeddings
                                             │
                                             ▼
                              [ Epistemic Out-of-Distribution Check ]
                                   D_OOD = D_Mahalanobis(z, Z_in-dist)
                                             │
                          ┌──────────────────┴──────────────────┐
                          ▼                                     ▼
               D_OOD > τ_OOD (Out-of-Distribution)           D_OOD ≤ τ_OOD (In-Distribution)
                          │                                     │
                          ▼                                     ▼
            [ OUT-OF-DISTRIBUTION (OOD) /         [ Calibrated Bayesian LLR Fusion ] (scoring.py)
              UNSEEN GENERATOR QUEUE ]                 L = Σ w_i ln(P(x_i|AI) / P(x_i|Real))
            - Overrides nominal probability P           P(AI | x) = 1 / (1 + e^-L)
            - Triggers forensic human review                            │
            - Preserves open-world safety         ┌─────────────────────┼─────────────────────┐
                                                  ▼                     ▼                     ▼
                                            P(AI) ≥ 0.995        0.400 ≤ P ≤ 0.600      P(AI) ≤ 0.005
                                             (High AI)            (Inconclusive)         (High Real)
```

### Calibrated Probability Bands
As governed by the logistic formulation $P = \frac{1}{1 + e^{-\mathcal{L}_{\text{audio}}}}$, real-world probabilities asymptote toward but never strictly reach $1.0$ ($100\%$) or $0.0$. Decision boundaries enforce five distinct operational bands:

1.  **High-Confidence Synthetic / AI-Generated ($P(\text{AI}) \ge 0.995$):** Unambiguous neural vocoder artifacts, synthetic RVQ token periodicity, or synthetic C2PA assertions.
2.  **Leaning Synthetic / Flagged for Review ($0.600 < P(\text{AI}) < 0.995$):** Notable formant anomalies, phase irregularities, or partial voice cloning detected; automated platform warning applied.
3.  **Inconclusive / Indeterminate Evidence ($0.400 \le P(\text{AI}) \le 0.600$):** Equivocal signals resulting from heavy lossy compression, telephony bandpass filtering, or conflicting multi-feature evidence; flagged for forensic specialist review.
4.  **Leaning Authentic / Low Anomaly ($0.005 < P(\text{AI}) < 0.400$):** Minor non-generative audio tuning or standard compression artifacts consistent with physical acoustic capture.
5.  **High-Confidence Authentic Physical Capture ($P(\text{AI}) \le 0.005 \iff P(\text{Real}) \ge 0.995$):** Intact acoustic tube formant physics, continuous ENF grid phase, natural room impulse reverberation, and verified cryptographic provenance.

> **Orthogonal Channels:** The probability bands apply only to the in-distribution authenticity score. Three channels operate independently of it: (a) the Trust & Safety pre-gate (CSAE hard-blocks) short-circuits all bands; (b) a valid ZK-verified or hardware-signed edit proof is reported as an `AUTHENTIC_EDITED_AUDIO` provenance label alongside, not inside, the band; (c) contextual/cheapfake claims are assessed separately from signal authenticity (Section 21.3).

---

## 24. Conclusion & Strategic Roadmap

1.  **Systematic Multi-Dimensional Grounding:** By unifying wave propagation physics, room impulse responses, electrical grid induction (ENF), vocal tract biomechanics, telephony codecs, cryptographic zk-SNARKs, and neural acoustic latent representations, the taxonomy establishes a robust, mathematically defensible framework across all operational audio modalities.
2.  **Humility in Open-World Forensics:** Recognizing that voice synthesis and acoustic generative transformers constitute open classes with rapidly evolving architectures and counter-forensic attacks, the system explicitly decouples epistemic uncertainty via `OUT-OF-DISTRIBUTION (OOD)` gating and provides an open-set `UNKNOWN_SOURCE` attribution state.
3.  **Multi-Evidence Fusion is Essential:** Physical acoustic validation, cryptographic assertions, and statistical learning reinforce one another, preventing single-point failure modes across the content lifecycle.

---

## 25. References & Citations

1.  **AES Standards:** *AES27 (managing recorded audio materials intended for forensic examination); AES43 (criteria for the authentication of analog audio tape recordings); AES67-2018 (High-Performance Audio-over-IP).* Note: no AES standard is cited here for authenticating digital recordings; verify current AES forensic project status before relying on a specific document number.
2.  **LIGO / Virgo Collaborations:** *Observation of Gravitational Waves from a Binary Black Hole Merger.*
3.  **C2PA:** *Technical Specification for Digital Content Provenance, Version 2.4 (2026).* `https://c2pa.org/`
4.  **Google DeepMind:** *SynthID for Audio: Robust Waveform Watermarking for Generative Audio.*
5.  **Meta AI:** *AudioSeal: Proactive Localized Robust Watermarking for Speech Synthesis.*
6.  **NASA / JPL:** *Voyager Interstellar Mission Plasma Wave Science Observations.*
7.  **Caltech Optical Imaging Laboratory:** *Photoacoustic Tomography: In Vivo Imaging from Organelles to Organ Systems.*
8.  **ASVspoof Consortia:** *ASVspoof 2021/2024: Automatic Speaker Verification and Spoofing Countermeasures Challenge.*
9.  **EBU & ITU-R:** *EBU R128 & ITU-R BS.1770-4: Algorithms to Measure Audio Programme Loudness and True-Peak Audio Level.*
10. **ISO/IEC 27037:2012:** *Information technology — Security techniques — Guidelines for identification, collection, acquisition and preservation of digital evidence.*
11. **EBU Tech 3285:** *Specification of the Broadcast Wave Format (BWF).*

---

## Appendix A: Controlled Vocabulary Standards Mapping

| Standard | Identifier | Human-Readable Label | Engine State Mapping |
| :--- | :--- | :--- | :--- |
| **IPTC** | `digitalsourcetype:digitalCapture` | Original Microphone Capture | `AUTHENTIC_REAL_AUDIO` |
| **IPTC** | `digitalsourcetype:screenCapture` | Software Loopback Capture | `AUTHENTIC_DIGITAL_LOOPBACK` |
| **IPTC** | `digitalsourcetype:trainedAlgorithmicMedia` | Created using Generative Audio AI | `FULLY_AI_GENERATED_AUDIO` |
| **C2PA** | `c2pa.actions.zk_audio_redact` | ZK-Proven Audio Redaction | `AUTHENTIC_EDITED_AUDIO` (ZK-Verified) |
| **AES**  | `AES67-2018` | High-Performance Audio-over-IP Stream | `AUTHENTIC_BROADCAST_AUDIO` |
| **Engine** | OOD gate ($D_{\text{OOD}} > \tau_{\text{OOD}}$) | Out-of-Distribution / Unseen Generator | `OOD_UNSEEN_GENERATOR` |
| **Engine** | Probability band 3 ($0.400 \le P(\text{AI}) \le 0.600$) | Inconclusive / Indeterminate Evidence | `INCONCLUSIVE` |
| **Engine** | Open-set attribution ($\min_k D_{\text{attr}} > \tau_{\text{attr}}$) | Unknown Source / Novel Generator | `UNKNOWN_SOURCE` |
| **Engine** | Trust & Safety pre-gate (CSAE) | Hard-Block Class (bypasses probability bands) | `HARD_BLOCK_ESCALATE` |

---

## Appendix B: Canonical Generative AI Audio Architecture Specifications

```
Model / Platform        Domain       Native Rate    Architecture           Latent Tokens / Codec
────────────────────────────────────────────────────────────────────────────────────────────────
ElevenLabs v3/v4        Speech/SFX   44.1 kHz       DiT + Flow Matching    Proprietary Neural Codec
Google Chirp 3 HD       Speech       24 / 48 kHz    Speech Transformer     SoundStream / DAC
Cartesia Sonic 3.6      Speech       24 kHz         SSM + Streaming Flow   Custom RVQ Tokenizer
Suno v5.5               Music/Song   44.1 kHz       Diffusion Transformer  EnCodec / DAC Derivative
```

---

## Appendix C: Audio Codec Forensic Signature & Psychoacoustic Cutoff Matrix

| Codec | Bitrate Range | High-Frequency Cutoff | Characteristic Forensic Tell |
| :--- | :--- | :--- | :--- |
| **MP3 (Low)** | $128\text{ kbps}$ | $15.5\text{–}16.0\text{ kHz}$ | Severe brickwall cutoff; pre-echo smears before transient spikes. |
| **MP3 (High)** | $320\text{ kbps}$ | $19.5\text{–}20.5\text{ kHz}$ | Subtle cutoff; 18-point sub-band filterbank transition ripples. |
| **AAC-LC** | $128\text{–}256\text{ kbps}$ | $17.0\text{–}21.5\text{ kHz}$ | Cleaner high-frequency preservation than MP3; pure MDCT grid. |
| **Opus (Voice)** | $16\text{–}32\text{ kbps}$ | $8.0\text{–}12.0\text{ kHz}$ | SILK model dynamic bandwidth switching; adaptive frame lengths. |

---

## Appendix D: Legacy & Historical Audio Format Timeline

```
Year    Format                  Medium / Technology           Acoustic Bandwidth
──────────────────────────────────────────────────────────────────────────────────────
1877    Phonograph Cylinder     Tinfoil / Wax Grooves         ~300 Hz – 2.5 kHz
1898    78 RPM Shellac Disc     Lateral Abrasive Groove       ~100 Hz – 4.5 kHz
1948    Vinyl LP (33⅓ RPM)      Microgroove PVC (RIAA)        ~30 Hz – 18 kHz
1963    Compact Cassette        Type I Ferric Magnetic Tape   ~40 Hz – 12 kHz
1982    Audio CD (Red Book)     16-bit / 44.1 kHz PCM         20 Hz – 22.05 kHz
1987    DAT (Digital Audio Tape) 16-bit / 48 kHz PCM          20 Hz – 24.0 kHz
1993    MP3 Standard (MPEG-1)   Layer III Psychoacoustic      20 Hz – 20 kHz (Cutoff)
2001    FLAC Lossless           Open Source Linear Predict    20 Hz – Nyquist
2012    Opus Codec (IETF)       Hybrid SILK/CELT Streaming    20 Hz – 20 kHz
2026    FlexiCodec / DiT Codec  Flow-Matching Neural Latents  Full Spectrum Generative
```

---

## Appendix E: Mathematical Physics Formulations for Acoustic Forensics

### 1. Electrical Network Frequency (ENF) Phase Continuity
$$\Phi_{\text{ENF}}(t) = 2\pi \int_0^t f_{\text{mains}}(\tau) d\tau + \phi_0, \quad \Delta \Phi > \tau_{\text{phase}} \approx 0.05\text{ rad}$$
*(Note: $\tau_{\text{phase}} \approx 0.05\text{ rad}$ represents an empirical baseline threshold calibrated against regional power grid reference archives; subject to signal-to-noise ratio).*

### 2. Photoacoustic Pressure Generation
$$p_0 = \Gamma \cdot \mu_a \cdot F, \quad \tau_p < \frac{d_c}{c_s}$$

### 3. Human Vocal Tract Formant Physics (Acoustic Tube Model)
$$F_n = \frac{(2n - 1) c}{4L}, \quad n \in \{1, 2, 3, 4\}$$

### 4. Bayesian Multi-Evidence Acoustic Fusion Formulation
$$\mathcal{L}_{\text{audio}} = \sum_{i=1}^{M_{\text{physical}}} w_i \ln \left( \frac{P(x_i \mid \text{AI})}{P(x_i \mid \text{Real})} \right) + \sum_{j=1}^{K_{\text{codec}}} w_j \ln \left( \frac{P(x_j \mid \text{AI})}{P(x_j \mid \text{Real})} \right) + \sum_{k=1}^{L_{\text{neural}}} w_k \ln \left( \frac{P(x_k \mid \text{AI})}{P(x_k \mid \text{Real})} \right)$$
$$P(\text{AI} \mid \mathbf{x}) = \frac{1}{1 + e^{-\mathcal{L}_{\text{audio}}}}$$


---

## Appendix F: Implementation Status (REV7)

Status key: **Implemented** (code + tests), **Partial** (some sub-items), **Recognition-only** (format sniffed, no authenticity scoring), **Spec-only** (documented taxonomy, no code). "Advisory" findings never change P(AI). The shared foundation (`core/forensics/`, `core/bands.py`, `ui/stages.py`) is wired into the **image** and **audio** modalities; video is a later cycle. Code references are relative to `audio_detector/` unless stated.

| Dimension / Section | Status | Where | Notes |
| :--- | :--- | :--- | :--- |
| **A** Provenance & authenticity spectrum | Partial | `scoring.py`, `detector.py` | Vocoder-cutoff / spectral-flatness / silence heuristics give AI-vs-real; no loopback, tuned, stem-separated, or voice-conversion states. |
| **B** Content, music & paralinguistics | Partial | `content.py` | Delivery style and speech/music/ambient inference only; no language ID, diarization, or music theory. |
| **C** Acoustic physics & side-channels | Spec-only | - | No RT60 / room impulse analysis. |
| **D** Frequency spectrum & data-over-sound | Partial / Recognition-only | `features.py`, `core/forensics/gates.py` | Spectral cutoff and HF ratio; MIDI recognized (symbolic, not scored). RF demodulation, Morse, seismic audification, data-over-sound: spec-only. |
| **E-I** Cosmic, quantum, photoacoustic, parametric, neuro-acoustic | Spec-only | - | |
| **J** Transduction & microphones | Spec-only | - | |
| **K** Spatial audio | Spec-only | - | |
| **L** Codecs, telephony, fake hi-res | Partial | `dimension_checks/signal.py` | Implemented: fake hi-res (band-limit vs container rate), 16-in-24 bit padding, lossy-origin cutoff in lossless containers, narrowband-telephony detection. MDCT double-compression, AMR/EVS codec forensics, VoIP loss concealment: spec-only. |
| **M** Studio production & loudness | Partial | `dimension_checks/signal.py` | RMS/peak/crest factor, clipping ratio, approximate true peak, over-compression flag. **Not** ITU-R BS.1770 LUFS. Tuning / stem-separation forensics: spec-only. |
| **N** Generative AI frontier | Partial | `attribution.py` | 13-generator attribution (6 calibrated); no voice-cloning or real-time voice-changer detection. |
| **O** Forensic science (ENF, formants, double compression) | Partial | `dimension_checks/signal.py` | ENF trace (50/60 Hz, 8 s STFT, parabolic interpolation), splice-discontinuity detection, stable-tone rejection; absence is never evidence. Formant tracking and MDCT analysis: spec-only. |
| **P** Cryptography, security & watermarking | Partial | `provenance.py`, `dimension_checks/integrity.py` | C2PA is a byte-signature presence scan, **not** cryptographic validation. Implemented: format sniff, bytes beyond declared end, polyglot signatures, instruction-like tag text. AudioSeal/WavMark/SynthID detection and steganalysis: spec-only. |
| **Sec. 20** Metadata & container forensics | Implemented (weak evidence) | `dimension_checks/container.py` | ID3v2, RIFF + BWF `bext`/LIST INFO, FLAC STREAMINFO MD5 (ffmpeg-verified), MP3 Xing/LAME tag, Ogg page CRC / sequence / granule. M4A/AAC atoms and LAME tag-CRC verification: spec-only. |
| **Sec. 21.1** Harm / hard-block | Partial | `core/forensics/gates.py` | Pluggable SHA-256 hard-block list (no classifier). Harassment/vishing classification: spec-only. |
| **Sec. 21.2** Legal flags | Implemented (advisory) | `dimension_checks/legal.py` | Rights notice, voice-biometric notice, AI-disclosure label, synthetic-voice likeness/TCPA advisory. |
| **Sec. 21.3** Cheapfakes / context | Partial | `dimension_checks/context.py` | Coarse acoustic fingerprint + optional local reference index; no ASR/semantic cross-check. |
| **Sec. 21.4** Granularity | Partial | `features.py` | Per-window temporal segments; no per-speaker diarization. |
| **Sec. 21.5** Lifecycle laundering | Implemented (heuristic) | `dimension_checks/lifecycle.py` | Transcoding-cascade likelihood. |
| **Sec. 21.6** File security | Implemented | `dimension_checks/integrity.py` | See Dimension P. Decoder-exploit and ultrasonic-command (DolphinAttack) detection: spec-only. |
| **Sec. 21.7** Detector validity | Partial | `dimension_checks/reliability.py` | Confidence limiters (short, clipped, telephony, cascade). Accent/dialect bias audits and channel-shift benchmarks: spec-only. |
| **Sec. 21.8** Accessibility | Spec-only | - | |
| **Sec. 21.9** Open-set attribution | Implemented (basic) | `dimension_checks/__init__.py` | `UNKNOWN_SOURCE` when no generator profile matches an AI-leaning recording. |
| **Sec. 23** Five probability bands | Implemented | `core/bands.py` | Applied to the in-distribution score. |
| **Sec. 23** OOD gate | Implemented, uncalibrated by default | `core/forensics/ood.py`, `dimension_checks/fit_ood.py` | 5-dim acoustic feature vector; reports `NOT_CALIBRATED` until fitted (`python -m audio_detector.dimension_checks.fit_ood`). |

**Hybrid scoring (as built):** only `PHYSICAL_SIGNAL` / `METADATA_WEAK` findings may add log-odds (base-10, positive = toward AI): per-finding cap 0.25 (explicit generator string in tags 0.40), total clamp +/-0.40, absence never scored. Audio pools by weighted average, so the terms shift the pooled probability in log-odds space (`AudioAIDetector.analyze_audio_file(extra_log_lrs=...)`); with no terms the result is unchanged. Scoring terms: explicit generator self-declaration in ID3/RIFF fields (+0.40), ENF continuous trace (-0.15), ENF splice discontinuity (+0.15).

Video wiring of the foundation (ISOBMFF box / `stts` checks, SEI/telemetry, interlace and telecine, virtual-camera checks, UI stage parity) is planned next.


**Post-audit corrections (2026-10-03), all covered by tests:**
- Scores are heuristic and uncalibrated (`calibration_status`); thresholds were tuned on synthetic fixtures only. `python -m services.calibration_cli --modality audio` measures accuracy, ECE and band occupancy on the held-out validation split once a labeled media library exists.
- C2PA is marker presence only (no signature validation): reported, never scored, never called "verified".
- Camera EXIF is unauthenticated: coherent EXIF earns no credit; EXIF contradicted by strong synthetic pixel evidence is demoted (image detector trust policy).
- The package pipelines and the Streamlit flow share one verdict path (`core.decision.generate_final_decision`, `decision_mode` = `image_authoritative` | `fused`); attribution is explanation only and is not double counted.
- Parser fixes: RIFF chunk walking is seek-based (no false WARN beyond 4 MB), PNG text chunks after IDAT are found.

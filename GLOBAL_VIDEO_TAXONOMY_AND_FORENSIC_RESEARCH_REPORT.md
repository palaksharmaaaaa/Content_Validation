# Comprehensive Global Video Taxonomy, Universal Temporal Spectrum, and Generative AI Video Synthesis Architecture

**Document Reference:** `RPT-VID-FOR-2026-OCT-02-REV7`  
**Referenced Standards (informational only; this software is not certified or audited against them):** NIST OpenMFC, C2PA Technical Specification v2.4, IPTC Video Metadata Standard (2024–2026), ISO/IEC 27037:2012, SMPTE ST 2110, ITU-R BT.2100, DICOM PS3.1-2026 (clinical video)  
**Classification:** Technical Architecture & Omnidimensional Video Forensic Specification  
**Publication Date:** October 2, 2026  
**Timestamp:** 2026-10-02  
**Repository Working Directory:** `<repo>`  
**Author:** Antigravity Advanced Agentic Coding & Video Forensic Engineering Team  

---

## Revision History

| Revision | Date / Timestamp | Scope & Key Modifications |
| :--- | :--- | :--- |
| **REV1** | 2026-10-02T11:45:00+05:30 | Initial video taxonomy draft; baseline temporal features and compression formats. |
| **REV2** | 2026-10-02T12:30:00+05:30 | Expanded astrophysical, femtosecond SCARF, and neuromorphic DVS video streams. |
| **REV3** | 2026-10-02T13:22:00+05:30 | Added virtual camera attacks, interlacing, streaming protocols, telemetry, world models, avatars, and watermarking. |
| **REV4** | 2026-10-02T14:20:00+05:30 | **Production Forensic Integrity Release:** Enforced strict 16-dimension ontology (A–P); added Section 20 Dedicated Metadata & Container-Structure Forensics (ISOBMFF box order, NAL SEI strings, telemetry tracks); added Doppler radar and sonar continuous video; added Stereoscopic 3D & MV-HEVC multi-view; expanded watermarking landscape (SynthID, Digimarc, IMATAG, Truepic, C2PA); codified legal/regulatory standards (Texas CUBI, Regulation (EU) 2024/1689 Art. 50, ITAR/EAR $<0.3\text{ m}$ GSD, HIPAA/EHDS surgical video PHI); integrated detector demographic validity (Fitzpatrick I–VI) and accessibility tracks (CEA-608/708, WebVTT); calibrated PRNU to VISION and Dresden benchmarks with "Absent $\ne$ AI" physical safeguards (global shutter CMOS, temporal noise reduction); decoupled epistemic OOD detection from Bayesian probability bands; and established the 5 calibrated operational probability bands. |
| **REV5** | 2026-10-02 | **Consistency Pass:** Removed off-topic IEEE 3333.1 and non-video FITS citations; synced ontology lines for Dimensions D and L; fixed TOC anchor for Section 21; corrected MPEG-TS continuity-counter wording; added metadata evidentiary-weight caveat; added engine-state rows to Appendix A; added orthogonal-channel note; clarified that pipeline stages are target-architecture. |
| **REV6** | 2026-10-02 | **Status Release:** Added Appendix F (implementation status per dimension). The shared dimension-check foundation exists in `core/` (wired into image first); video wiring is pending. |
| **REV7** | 2026-10-02 | **Implementation Release (video):** The video pipeline now implements the shared dimension-check foundation: hard-block and scientific-format gates; file-integrity checks; ISOBMFF box/sample-table/metadata/telemetry/encoder-string and EBML container checks; interlacing and frame-cadence signal checks; fingerprint/legal/lifecycle/reliability advisory stages; five probability bands, OOD gate (uncalibrated until fitted) and open-set `UNKNOWN_SOURCE`. Findings, band and reliability appear on the Evidence tab of the video result page (the earlier numbered Stage 1b/3b/4b/6b layout was replaced by a verdict card with Overview, Evidence, Details and Feedback tabs). Appendix F rewritten; the foundation is now wired into image, audio and video. |

---

## Executive Summary

The emergence of Spatio-Temporal Video Diffusion Transformers (VideoDiT), deterministic flow-matching temporal architectures, 4D Gaussian Splatting (4DGS), real-time neural lip-sync re-enactment, interactive generative world models, virtual camera deepfake injection, asynchronous neuromorphic event video streams, single-shot femtophotography (SCARF at $156.3\text{ trillion frames per second}$), Zero-Knowledge Proofs for verifiable temporal video editing (zk-SNARKs), and adversarial counter-forensics as of late 2026 has made elementary binary classification (*"Real vs. AI Video"*) obsolete.

A modern digital video is not merely a sequence of 2D images or a container file; it is an **open-class, multi-dimensional spatio-temporal physical, biological, optical, and computational signal** defined across sixteen operational axes:
1.  **Dimension A: Provenance, Origin, & Authenticity Spectrum:** Ground-truth physical camera sensor capture through multi-frame computational fusion, screen recording, neural inpainting, and fully synthetic generative video models.
2.  **Dimension B: Content Genre & Semantic Subject-Matter Catalog:** Narrative cinema, broadcast journalism, CCTV surveillance, micro-content social feeds, sports broadcast, endoscopic surgery, and defense reconnaissance.
3.  **Dimension C: Visual Art Styles, Cinematographic Grammar, & Animation Mechanics:** Camera kinetics, stabilization rigs, cel/stop-motion/anime animation, and real-time beautification AR facial warping filters.
4.  **Dimension D: Electromagnetic Spectrum, Multi-Spectral, Radar/Sonar, & Depth Video:** Visible, NIR/SWIR/MWIR/LWIR thermal, UV fluorescence, continuous fluoroscopy, continuous Doppler radar, acoustic sonar video, and synchronized RGB-D streams.
5.  **Dimension E: Astrophysical, Cosmic, & Interstellar Temporal Video:** Dynamic Event Horizon Telescope (EHT) accretion flow movies, Solar Dynamic Observatory (SDO) coronal mass ejections, and Mars rover supersonic descent dynamics.
6.  **Dimension F: Nanoscale, Subatomic, Ultra-High-Speed, & Quantum Video:** Single-shot femtosecond imaging (T-CUP, SCARF at $156.3\text{ trillion fps}$), in-situ DTEM electron microscopy, high-speed biological AFM (50 fps), and time-resolved SPAD arrays.
7.  **Dimension G: Optical Wavefront Physics, Polarization, & Aerodynamic Video:** Dynamic Stokes polarization streams, high-speed schlieren shockwave refractometry ($10^6\text{ fps}$), and digital holographic microscopy.
8.  **Dimension H: Geospatial, Terrestrial, & Planetary Remote Sensing Video:** Spaceborne LEO high-resolution video constellations ($<0.3\text{ m}$ GSD), Circular Video Synthetic Aperture Radar (VideoSAR GMTI), and 4D continuous LiDAR.
9.  **Dimension I: Neuromorphic Event-Based Temporal Streams:** Frame-free asynchronous Dynamic Vision Sensors (DVS/DAVIS) capturing continuous microsecond event clouds $(x, y, t, p)$ with $>120\text{–}140\text{ dB}$ dynamic range.
10. **Dimension J: Neuroimaging & BCI Continuous Video Decoding:** Direct decoding and continuous temporal video synthesis from human visual cortex BOLD hemodynamics via 7T ultrafast fMRI (Mind-Video).
11. **Dimension K: Temporal Dynamics, Interlacing, Frame Rates, & Motion Mechanics:** Frame rates ($1\text{ to }1.56\times 10^{14}\text{ fps}$), interlacing (TFF/BFF), 3:2 telecine pulldown, rolling shutter vs global shutter CMOS kinematics, and AI frame interpolation.
12. **Dimension L: Spatial Geometry, Stereoscopic 3D, Multi-View, & Immersive Volumetric Video:** Planar rectilinear, anamorphic squeeze, stereoscopic 3D, Multi-View Video Coding (MVC / MV-HEVC), VR180/360° equirectangular, and dynamic 6-DoF 4D Gaussian Splatting.
13. **Dimension M: Technical Codecs, Streaming Transport, Telemetry, & Containers:** Compression codecs (H.264, H.265, H.266/VVC, AV1, ProRes RAW), streaming transports (HLS, DASH, WebRTC), and embedded hardware IMU/GPS telemetry.
14. **Dimension N: Audio-Visual Synchronization & Multimodal Binding:** Phoneme-viseme temporal lip sync ($|\Delta t| < 40\text{ ms}$), voice biometric facial concordance, and room acoustic impulse reverberation matching.
15. **Dimension O: Generative AI Video, World Models, Neural Avatars, & Counter-Forensics:** Foundation flow-matching VideoDiT engines (Kling 3.0, Veo 3.1, Gen-4.5), interactive real-time world simulators, talking-head neural avatars, and anti-forensic PRNU injection attacks.
16. **Dimension P: Video Forensics, Watermarking, Cryptography, & Steganography:** Zero-Knowledge video redaction (zk-SNARKs), multi-layer video watermarking (SynthID, Digimarc, IMATAG, Truepic, C2PA), motion vector steganography, and ViT deep steganalysis.

This document establishes the operational forensic taxonomy of video categories across the physical universe, biological organisms, broadcast/streaming platforms, and generative frontiers as of October 2026. It formalizes **sixteen analytical dimensions**, introduces a comprehensive **Cross-Cutting Legal, Safety, & File Security Framework**, establishes mathematical temporal physics formulations, and defines the calibrated evidence fusion architecture for the `project-content-validation` engine.

---

## Table of Contents

1. [Inquiry Context & Research Mandate](#1-inquiry-context--research-mandate)
2. [Research Methodology & Primary Information Sources](#2-research-methodology--primary-information-sources)
3. [The Sixteen-Dimensional Video Taxonomy Ontology](#3-the-sixteen-dimensional-video-taxonomy-ontology)
4. [Dimension A: Provenance, Origin, & Authenticity Spectrum](#4-dimension-a-provenance-origin--authenticity-spectrum)
5. [Dimension B: Content Genre & Semantic Subject-Matter Catalog](#5-dimension-b-content-genre--semantic-subject-matter-catalog)
6. [Dimension C: Visual Art Styles, Cinematographic Grammar, & Animation Mechanics](#6-dimension-c-visual-art-styles-cinematographic-grammar--animation-mechanics)
7. [Dimension D: Electromagnetic Spectrum, Multi-Spectral, Radar/Sonar, & Depth Video](#7-dimension-d-electromagnetic-spectrum-multi-spectral-radarsonar--depth-video)
8. [Dimension E: Astrophysical, Cosmic, & Interstellar Temporal Video](#8-dimension-e-astrophysical-cosmic--interstellar-temporal-video)
9. [Dimension F: Nanoscale, Subatomic, Ultra-High-Speed, & Quantum Video](#9-dimension-f-nanoscale-subatomic-ultra-high-speed--quantum-video)
10. [Dimension G: Optical Wavefront Physics, Polarization, & Aerodynamic Video](#10-dimension-g-optical-wavefront-physics-polarization--aerodynamic-video)
11. [Dimension H: Geospatial, Terrestrial, & Planetary Remote Sensing Video](#11-dimension-h-geospatial-terrestrial--planetary-remote-sensing-video)
12. [Dimension I: Neuromorphic Event-Based Temporal Streams](#12-dimension-i-neuromorphic-event-based-temporal-streams)
13. [Dimension J: Neuroimaging & BCI Continuous Video Decoding](#13-dimension-j-neuroimaging--bci-continuous-video-decoding)
14. [Dimension K: Temporal Dynamics, Interlacing, Frame Rates, & Motion Mechanics](#14-dimension-k-temporal-dynamics-interlacing-frame-rates--motion-mechanics)
15. [Dimension L: Spatial Geometry, Stereoscopic 3D, Multi-View, & Immersive Volumetric Video](#15-dimension-l-spatial-geometry-stereoscopic-3d-multi-view--immersive-volumetric-video)
16. [Dimension M: Technical Codecs, Streaming Transport, Telemetry, & Containers](#16-dimension-m-technical-codecs-streaming-transport-telemetry--containers)
17. [Dimension N: Audio-Visual Synchronization & Multimodal Binding](#17-dimension-n-audio-visual-synchronization--multimodal-binding)
18. [Dimension O: State of the Art in Generative AI Video, World Models, & Counter-Forensics](#18-dimension-o-state-of-the-art-in-generative-ai-video-world-models--counter-forensics)
19. [Dimension P: Video Forensics, Watermarking, Cryptography, & Steganography](#19-dimension-p-video-forensics-watermarking-cryptography--steganography)
20. [Section 20: Dedicated Metadata, Container-Structure, & Telemetry Forensics](#20-dedicated-metadata-container-structure--telemetry-forensics)
21. [Section 21: Cross-Cutting Framework (Legal, Safety, Cheapfakes, Granularity, Security)](#21-cross-cutting-framework-legal-safety-cheapfakes-granularity--security)
22. [Section 22: Forensic Physics & Mathematical Differentiation Formulations](#22-forensic-physics--mathematical-differentiation-formulations)
23. [Section 23: System Architectural Integration & Calibrated Bayesian Verification](#23-system-architectural-integration--calibrated-bayesian-verification)
24. [Section 24: Conclusion & Strategic Roadmap](#24-conclusion--strategic-roadmap)
25. [Section 25: References & Citations](#25-references--citations)
26. [Appendix A: Controlled Vocabulary Standards Mapping](#appendix-a-controlled-vocabulary-standards-mapping)
27. [Appendix B: Canonical Generative AI Video Model Specifications](#appendix-b-canonical-generative-ai-video-model-specifications)
28. [Appendix C: Video Codec Forensic Signature & Compression Matrix](#appendix-c-video-codec-forensic-signature--compression-matrix)
29. [Appendix D: Legacy & Historical Video Format Timeline](#appendix-d-legacy--historical-video-format-timeline)
30. [Appendix E: Mathematical Physics Formulations for Temporal Forensics](#appendix-e-mathematical-physics-formulations-for-temporal-forensics)
31. [Appendix F: Implementation Status (REV7)](#appendix-f-implementation-status-rev7)

---

## 1. Inquiry Context & Research Mandate

The objective of this specification is to establish a rigorous, mathematically defensible, and comprehensive operational video taxonomy for the `project-content-validation` platform as of late 2026.

Modern digital video represents an open-class problem: video sensors, compression formats, neural diffusion pipelines, and adversarial manipulation vectors evolve constantly. While no closed system can claim absolute zero-omission over an unbounded future, this report provides exhaustive coverage across known optical, broadcast, scientific, computational, and generative video systems, formalizing explicit handling for `UNKNOWN`, `AMBIGUOUS`, and `OUT-OF-DISTRIBUTION (OOD)` generative models.

---

## 2. Research Methodology & Primary Information Sources

Information was retrieved, cross-validated, and compiled from the following bodies:
*   **Standards Bodies:** C2PA v2.4, SMPTE ST 2110, ITU-R BT.2100, ISO/IEC 27037:2012, NIST OpenMFC, IPTC Video NewsCodes, W3C Media Source Extensions.
*   **Frontier Laboratories & Industry Consortia:** INRS Ultrafast Laboratory (SCARF at $156.3\text{T fps}$), Event Horizon Telescope Collaboration, Prophesee/Sony IMX636 event sensors, Mind-Video fMRI decoding consortia, and verifiable video zk-SNARK engineering frameworks.

---

## 3. The Sixteen-Dimensional Video Taxonomy Ontology

```
[ Dimension A: Provenance & Authenticity ]   ──> Physical optical capture, edit, hybrid, AI synthesis, virtual camera injection.
[ Dimension B: Content Genre & Subject ]      ──> Cinema, documentary, broadcast news, surveillance, medical, UGC.
[ Dimension C: Visual Art & Cinematography ] ──> Camera movement, shot scale, film movement, AR filters, animation style.
[ Dimension D: Electromagnetic & Depth Video]──> Visible RGB, IR, UV, fluoroscopy, LiDAR/ToF depth, Doppler radar, sonar video.
[ Dimension E: Astrophysical & Interstellar ]──> EHT black hole accretion movies, SDO solar flare video, Mars EDL.
[ Dimension F: Nanoscale, Ultrafast, Quantum]──> Femtosecond SCARF (156T fps), in-situ DTEM atomic movies, HS-AFM.
[ Dimension G: Optical Wavefront & Pol ]     ──> High-speed Stokes polarimetry, Schlieren aerodynamics, DHM.
[ Dimension H: Geospatial & Remote Sensing ] ──> LEO satellite video constellations, VideoSAR GMTI, LiDAR swaths.
[ Dimension I: Neuromorphic Event Streams ]  ──> Asynchronous frame-free DVS/DAVIS event clouds (x, y, t, p).
[ Dimension J: BCI Continuous Video Decoding]──> Continuous visual cortex fMRI decoding (Mind-Video).
[ Dimension K: Temporal Dynamics & Interlace]──> Frame rates (1 to 156T fps), shutter angles, interlacing (480i/1080i, 3:2 pulldown).
[ Dimension L: Spatial Geometry & 3D / VR ]  ──> 2D flat, anamorphic 2x, stereo 3D / MV-HEVC, 360° VR, 6-DoF, 4DGS.
[ Dimension M: Codecs, Streaming, & Telemetry]─> Codecs (H.266, AV1, ProRes), HLS/WebRTC streaming, embedded telemetry (GPS/gyro).
[ Dimension N: Audio-Visual Synchronization ]──> Phoneme-viseme lip-sync, voice biometrics, room acoustic binding.
[ Dimension O: Generative AI & World Models ]──> VideoDiT flow matching, interactive world models, avatar lip-sync.
[ Dimension P: Forensics, Watermarks & Stego ]─> Temporal PRNU, zk-SNARK trimming proofs, SynthID watermarks, MV steganography.
```

---

## 4. Dimension A: Provenance, Origin, & Authenticity Spectrum

1. **State 1: Pure Physical Optical Sensor Capture (`digitalCapture`):** CMOS/CCD continuous capture, frame-to-frame PRNU correlation ($\rho \ge 0.15$), rolling shutter dynamics.
2. **State 2: Computational Video Capture (`computationalCapture`):** Multi-frame DSP merge, HDR+ video, Night Sight video.
3. **State 3: Analog Cine Film & Magnetic Tape (`negativeFilm`, `print`):** 35mm/70mm silver halide grain, gate weave, VHS/Betacam dropout streaks.
4. **State 4: Display Screen Recapturing:** Temporal Moiré refresh banding, subpixel grid magnification, keystone distortion.
5. **State 5: Digital Framebuffer Capture (`screenCapture`):** GPU desktop capture, zero sensor noise ($\sigma < 0.20$), Variable Frame Rate (VFR).
6. **State 6: Conventional NLE Editing (`minorHumanEdits`, `compositeSynthetic`):** Adobe Premiere/DaVinci cuts, chroma key spill, GOP resets.
7. **State 7: Procedural Raytraced 3D CGI (`softwareImage`):** Unreal Engine 5 Nanite, Blender Cycles raytracing, zero sensor noise.
8. **State 8: AI-Enhanced / Composite (Mix) (`compositeWithTrainedAlgorithmicMedia`):** Deepfake face swaps, neural lip-sync, temporal inpainting.
9. **State 9: Fully AI-Generated Video (`trainedAlgorithmicMedia`):** Kling 3.0, Veo 3.1, Runway Gen-4.5, zero physical PRNU, flow anomalies.
10. **State 10: Adversarially Spoofed / Anti-Forensic Video:** Injected synthetic PRNU, artificial rolling shutter wobble, stripped C2PA.
11. **State 11: Virtual Camera Injection Attacks:** Real-time deepfakes injected via synthetic DirectShow / v4l2loopback / OBS Virtual Camera drivers into WebRTC sessions (KYC onboarding spoofing, executive impersonation video calls), characterized by missing physical UVC hardware descriptors and driver-level framebuffer timestamp quantization.

---

## 5. Dimension B: Content Genre & Semantic Subject-Matter Catalog

*   **Narrative Cinema & Fiction:** Feature films, short films, web series, multi-camera setups, 24 fps cinematic cadence.
*   **Documentary & Non-Fiction:** Observational, expository, nature/wildlife, investigative journalism, interview talking-head framing.
*   **Broadcast News & Live Events:** News desk, press conferences, weather forecasts, live multi-camera switching, lower-third graphics.
*   **Music & Performance:** Music videos, concert multicam, theater (NT Live), opera, precise audio-visual synchronization.
*   **Social Media & Micro-Content:** TikTok/Shorts/Reels (9:16 vertical), ephemeral Stories, vlogs, reaction videos, ASMR, unboxing.
*   **Education & Technical Demos:** Video essays, lecture recordings, coding screencasts, annotation overlays.
*   **Gaming & Interactive:** Let's Plays, walkthroughs, esports feeds, game trailers, GPU framebuffer capture (60–120 fps).
*   **Surveillance & Public Safety:** CCTV fixed-camera, body-worn cameras (Axon Body 4), dashcams, continuous 24/7 loop recording.
*   **Sports & Athletic Motion:** Multi-camera sports feeds, ultra-slow-motion replays (120–480 fps), referee review (VAR).
*   **Medical & Clinical Endoscopy:** Laparoscopic surgery, endoscopy/colonoscopy, robotic surgery (Da Vinci), cine angiography.
*   **Scientific & Laboratory:** High-speed ballistics, shockwave schlieren, optical microscopy video.
*   **Industrial & Machine Vision:** Automated optical inspection (AOI), infrastructure drone scans, manufacturing monitoring.
*   **Defense & Aerospace:** Aerial reconnaissance, satellite surveillance, missile tracking telemetry, HUD cockpit video.

---

## 6. Dimension C: Visual Art Styles, Cinematographic Grammar, & Animation Mechanics

*   **Camera Movements:** Static tripod, Pan, Tilt, Dutch Angle, Whip Pan, Dolly In/Out, Truck Left/Right, Pedestal, Tracking Shot, Crane/Jib, Arc shot, Dolly Zoom ("Vertigo effect").
*   **Stabilization Rigs:** Steadicam, 3-axis motorized gimbals (DJI Ronin), Handheld physiological sway, Snorricam, Cablecam, FPV Racing Drones.
*   **Animation Paradigms:** Traditional Cel animation (animating on 1s or 2s), Stop-motion/claymation, Anime limited animation timing (on 2s, 3s, or 4s), Rotoscoping, Motion graphics & kinetic typography.
*   **AR Filters, Beautification, & Real-Time Retouching:** Real-time 3D facial landmark deformation, skin-smoothing bilateral/guided filtering (FaceTime, TikTok, Instagram, WeChat), synthetic eye enlargement, jaw slimming, and temporal boundary occlusion jitter.

---

## 7. Dimension D: Electromagnetic Spectrum, Multi-Spectral, Radar/Sonar, & Depth Video

*   **Visible Spectrum Video (400–700 nm):** Bayer RGGB/RYYB, True Monochrome cinema cameras, 3CMOS prism splitters.
*   **Infrared Video:** NIR night vision (850/940 nm), SWIR industrial inspection (1–3 µm), MWIR missile tracking (3–5 µm), LWIR radiometric FLIR thermography (8–14 µm).
*   **Ultraviolet Video (200–400 nm):** UV-A fluorescence video tracing mechanical leakages and crime scene fluids.
*   **Continuous High-Energy X-Ray Video (Fluoroscopy):** Real-time continuous X-ray video ($7.5\text{–}30\text{ fps}$) in angiography and barium swallowing studies.
*   **Continuous Time-of-Flight (ToF) & RGB-D Video:** Active IR modulated pulse streams (Apple TrueDepth, Azure Kinect, Intel RealSense) generating synchronized depth maps $D(x,y,t)$ alongside RGB streams; forensically distinguished by physical multipath interference and edge flight pixels absent in AI-synthesized monocular depth estimations.
*   **Continuous Doppler Radar & Acoustic Sonar Video:** Dynamic Doppler radar velocity video feeds (NEXRAD dual-polarization weather radar sequences, automotive FMCW millimeter-wave radar point clouds over time) and underwater acoustic video (Forward-Looking Sonar - FLS video, Dual-Frequency Identification Sonar - DIDSON, continuous side-scan acoustic video feeds); distinguished by acoustic reverberation decay, range-dependent beam spreading, and zero optical lens chromatic aberrations.

---

## 8. Dimension E: Astrophysical, Cosmic, & Interstellar Temporal Video

*   **Dynamic Radio Interferometry (EHT Sagittarius A*):** Time-resolved reconstruction algorithms synthesizing continuous video of orbiting plasma from sparse $u\text{-}v$ visibilities; machine learning frame forecasting (BHCast).
*   **Solar Dynamic Observatories (SDO & Solar Orbiter):** Continuous Extreme Ultraviolet (EUV) video ($171\text{ Å}, 193\text{ Å}, 304\text{ Å}$) capturing coronal mass ejections (CMEs) and magnetic reconnection.
*   **Spacecraft Entry, Descent, and Landing (EDL):** Perseverance Mars Rover high-speed video recording supersonic parachute deployment at Mach 1.7; Artemis SLS launch dynamics at $2,000\text{ fps}$.
*   **Relativistic Accretion Simulations (GRMHD):** Solving General Relativistic Magnetohydrodynamic equations visualizing Doppler boosting and relativistic jet launch mechanisms across temporal iterations.

---

## 9. Dimension F: Nanoscale, Subatomic, Ultra-High-Speed, & Quantum Video

*   **Single-Shot Femtosecond Video (T-CUP & SCARF):** Compressed Ultrafast Photography (T-CUP at $10\text{ trillion fps}$) and Swept-Coded Aperture Real-Time Femtophotography (SCARF at **$156.3\text{ trillion fps}$**) capturing laser pulses in flight, photonic Mach cones, and plasma bubble expansion.
*   **In-Situ Dynamic Transmission Electron Microscopy (DTEM):** Pulsed electron beam movies capturing atomic-scale chemical catalytic reactions, phase transitions, and crystal dislocation slips.
*   **High-Speed Biological AFM (HS-AFM):** Video rates of $30\text{–}50\text{ fps}$ visualizing molecular motors (myosin V) walking along actin filaments in physiological solutions.
*   **Time-Resolved SPAD Array 3D Video:** Picosecond photon timestamp streams for dynamic 3D LiDAR and Non-Line-of-Sight (NLOS) tracking around corners.

---

## 10. Dimension G: Optical Wavefront Physics, Polarization, & Aerodynamic Video

*   **High-Speed Stokes Polarization Video:** Real-time streams $\mathbf{S}(t) = [S_0(t), S_1(t), S_2(t), S_3(t)]^T$ visualizing dynamic mechanical stress birefringence propagation in polymers.
*   **High-Speed Schlieren & Shadowgraph Video:** Capturing dynamic refractive index gradients ($\nabla n(t)$) in supersonic bullet shockwaves, explosive detonations, and hypersonic wind tunnels at up to $1,000,000\text{ fps}$.
*   **Digital Holographic Microscopy (DHM) Video:** Interferometric video tracking micro-swimmers in 3D liquid volumes over time with numerical autofocusing.

---

## 11. Dimension H: Geospatial, Terrestrial, & Planetary Remote Sensing Video

*   **Spaceborne LEO Video Constellations:** SkySat and CarbonMapper recording sub-meter video sequences tracking urban traffic, vessel vectors, and invisible methane plume emissions.
*   **Circular Video Synthetic Aperture Radar (VideoSAR):** Continuous circular SAR tracks delivering real-time dynamic video ($1\text{–}5\text{ fps}$) with Ground Moving Target Indication (GMTI) through clouds and smoke.
*   **Airborne LiDAR Swaths:** Continuous 4D point cloud streams recording terrain deformation.

---

## 12. Dimension I: Neuromorphic Event-Based Temporal Streams

*   **Frame-Free Continuous Video:** Pixels operate asynchronously without a shutter, reporting logarithmic luminance changes:
    $$\Delta \ln(I) = \ln(I(x,y,t)) - \ln(I(x,y,t_{\text{prev}})) \ge \pm C_{\text{th}}$$
*   **Data Structure:** Continuous stream of tuples $e_i = (x_i, y_i, t_i, p_i)$ with microsecond timestamps and $> 120\text{–}140\text{ dB}$ dynamic range.
*   **Forensic Distinction:** Zero frames, zero rolling shutter, zero motion blur; paired with Spiking Neural Networks (SNN).

---

## 13. Dimension J: Neuroimaging & BCI Continuous Video Decoding

*   **fMRI Continuous Video Reconstruction (Mind-Video):** Using ultra-fast 7T fMRI (repetition times $\sim 125\text{ ms}$) to decode continuous BOLD hemodynamic signals in visual cortices, conditioning video diffusion models to synthesize reconstructed movie sequences of perceived or dreamed scenes.
*   **Forensic Boundary:** Reflects subjective neural perception, not an objective optical recording.

---

## 14. Dimension K: Temporal Dynamics, Interlacing, Frame Rates, & Motion Mechanics

*   **Frame Rate Spectrum:** 1–5 fps (CCTV) $\to$ 24 fps (Cinema standard, $180^\circ$ shutter) $\to$ 29.97/30/60 fps (Broadcast/Gaming) $\to$ 120–480 fps (Consumer slow-mo) $\to$ $10^3\text{–}10^6$ fps (Scientific high-speed) $\to$ $1.56 \times 10^{14}$ fps (SCARF).
*   **Interlacing & Telecine Forensics:**
    *   *Interlaced Scanning (480i, 576i, 1080i):* Two alternating fields per frame (Top Field First - TFF vs. Bottom Field First - BFF); spatial comb artifacts on horizontal motion; deinterlacing artifacts (bobbing flicker, weave ghosting, motion-adaptive interpolation residuals).
    *   *Telecine & 3:2 Pulldown:* Converting 24 fps film to 29.97 fps NTSC via a 3:2 field sequence ($A_1A_2, B_1B_2, B_1C_2, C_1D_2, D_1D_2$); cadence disruptions expose non-linear splicing in broadcast archives.
    *   *2:2 Pulldown:* Converting 24 fps to 25 fps PAL with a 4.1% audio-video speedup.
*   **Shutter Mechanics:** $180^\circ$ shutter rule ($t_{\text{exp}} = 1/(2 \cdot \text{FPS})$); Rolling shutter line-by-line CMOS skew vs. Global shutter instant exposure.
*   **Temporal Tampering:** Frame insertion, deletion, duplication, speed-ramping, and AI frame interpolation (RIFE/DAIN).

---

## 15. Dimension L: Spatial Geometry, Stereoscopic 3D, Multi-View, & Immersive Volumetric Video

*   **2D Planar & Anamorphic:** Standard rectilinear; Anamorphic ($1.33\times\text{–}2.0\times$ squeeze with oval bokeh).
*   **Stereoscopic 3D & Multi-View Video Coding (MVC / MV-HEVC):** Dual-channel binocular video (left/right eye with fixed interpupillary distance IPD $\approx 63\text{ mm}$), Multi-View Video Coding (ITU-T H.264 Annex H / MV-HEVC) exploiting both temporal and inter-view prediction redundancies; forensically validated via epipolar geometry constraints and horizontal disparity parallax consistency across stereo pairs.
*   **Immersive Spherical:** Equirectangular ($360^\circ \times 180^\circ$); VR180 stereoscopic 3D.
*   **Volumetric 6-DoF & 4D Gaussian Splatting (4DGS):** Millions of dynamic time-parameterized 3D Gaussian ellipsoids rasterized in real time ($> 100\text{ FPS}$).

---

## 16. Dimension M: Technical Codecs, Streaming Transport, Telemetry, & Containers

*   **Video Codecs:** H.264/AVC, H.265/HEVC, H.266/VVC (multi-type tree), AV1 (Film Grain Synthesis), Apple ProRes RAW, Blackmagic RAW, REDCODE RAW.
*   **Streaming & Transport Protocols:**
    *   *Adaptive Bitrate Streaming (HLS & MPEG-DASH):* Chunked delivery (`.m3u8` playlists, fragmented MP4 / `.m4s` segments); bitrate-switching discontinuities at segment boundaries.
    *   *Real-Time Interactive Video (WebRTC / RTP / RTCP):* UDP packet loss concealment, jitter buffer frame skips, macroblock freeze errors, and periodic intra-refresh slice smearing under network stress.
*   **Embedded Telemetry & Timecode Tracks:**
    *   *SMPTE Timecode:* Linear Timecode (LTC) and Vertical Interval Timecode (VITC); non-monotonic frame drops or discontinuities signal hidden temporal splices.
    *   *Hardware Telemetry Streams:* Action cam GPMF metadata (GoPro 3-axis accelerometer, gyroscope, GPS fix at 18 Hz), drone Flight Log subtitle tracks (DJI SRT/KML with altitude, motor speed, gimbal pitch), and Sony camera real-time lens focal/iris metadata.
*   **Color Pipelines:** Rec. 709, Rec. 2020, DCI-P3, ACES 2.0 (AP0/AP1), HDR10, HDR10+, Dolby Vision, HLG, camera Log curves (S-Log3, LogC4).

---

## 17. Dimension N: Audio-Visual Synchronization & Multimodal Binding

*   **Phoneme-Viseme Synchronization:** Temporal alignment between speech phonemes and visual visemes ($|\Delta t| < 40\text{ ms}$).
*   **Voice Biometrics:** Vocal tract resonant consistency with visible facial anatomy.
*   **Room Acoustic Concordance:** Reverberation matching visible room geometry.
*   **AI-Native Multimodal Synthesis:** Joint audio-visual token streams in VideoDiT models.

---

## 18. Dimension O: State of the Art in Generative AI Video, World Models, & Counter-Forensics

*   **Foundation Video Models:** Kling 3.0 / Omni (4K, 60fps), Google Veo 3.1 4K, Runway Gen-4.5, Seedance 2.5.
*   **Interactive Generative World Models:** Real-time diffusion world simulators (Oasis, GameNGen, Genie 2) generating interactive 3D physics environments from keyboard/controller inputs; characterized by lack of long-horizon temporal memory, drifting asset geometry, and hallucinated physics.
*   **Neural Avatars & Talking-Head Engines:** Video avatars (HeyGen, Synthesia, D-ID, SadTalker) driving synthetic human avatars from text scripts with automated lip-sync; exhibiting microscopic head pose jitter, teeth rendering uncanny valleys, and collar/neck occlusion blending flaws.
*   **Neural Manipulators:** LivePortrait face-swapping, Sync.so lip-sync re-enactment, Runway temporal inpainting.
*   **Temporal Counter-Forensics:** Synthetic PRNU noise injection across frames, fake rolling shutter wobble, C2PA manifest stripping.

---

## 19. Dimension P: Video Forensics, Watermarking, Cryptography, & Steganography

```
                                [ VIDEO CRYPTOGRAPHY, WATERMARKING & STEGANOGRAPHY ]
                                                         │
       ┌────────────────────────┬────────────────────────┴────────────────────────┬────────────────────────┐
       ▼                        ▼                                                 ▼                        ▼
 [ ZERO-KNOWLEDGE PROOFS ] [ VIDEO WATERMARKING ]                         [ MOTION VECTOR STEGO ]   [ DEEP VIDEO STEGANALYSIS ]
 • zk-SNARKs for Video     • DeepMind SynthID for Video                  • Sub-pixel Motion Vector • Spatio-Temporal ViT
 • Verifiable Trimming &   • Digimarc Barcode & IMATAG                     Modulation in H.265/VVC   (Long-Range Attention)
   Redaction Without Raw   • Hard vs Soft C2PA binding                   • Intra-Prediction Modes  • Residual Co-occurrence
```

### 19.1 Zero-Knowledge Proofs for Verifiable Video Redaction
*   Using **zk-SNARKs** to mathematically prove that a released video clip was derived via valid temporal trimming or pixelated face blurring from a cryptographically signed camera master without revealing redacted frames or private segments.

### 19.2 Multi-Layer Video Watermarking Landscape
*   **Google DeepMind SynthID-Video:** Latent-space temporal watermarking embedded directly during VideoDiT diffusion sampling; robust against downsampling, cropping, and frame interpolation.
*   **Digimarc Barcode for Video:** High-payload imperceptible spatial-temporal spread-spectrum patterns surviving re-encoding and broadcast distribution.
*   **IMATAG Video:** Micro-distortion forensic watermark impervious to platform compression and transcoding.
*   **Truepic & C2PA Hard/Soft Binding:** Cryptographic container binding (`uuid` ISOBMFF boxes) coupled with perceptual video hash assertions (e.g., PDQ-Video, TMK) guaranteeing provenance even when container metadata is stripped.

### 19.3 Video Steganography & Deep Steganalysis
*   **Motion Vector Perturbation:** Hiding data bits by subtly altering sub-pixel motion vectors in inter-frame prediction (P/B frames).
*   **Intra-Prediction Mode Modulation:** Modulating spatial prediction directions in I-frames to encode hidden payloads.
*   **Deep Video Steganalysis:** Spatio-Temporal Vision Transformers detecting non-local statistical disruptions across temporal feature maps.

---

## 20. Dedicated Metadata, Container-Structure, & Telemetry Forensics

Digital video container bitstreams and embedded telemetry provide structural signals that corroborate or refute pixel-level temporal continuity.

> **Evidentiary Weight:** Container layout, encoder strings, and telemetry are easily forged and are routinely rewritten or stripped by benign platforms, NLEs, and re-muxing tools. Box ordering (e.g., `moov` placement) and missing telemetry are weak heuristics, not proof of synthesis or editing. These signals enter the Bayesian LLR fusion as low-weight evidence and never as a sole determinant.

### 20.1 MP4 / ISOBMFF Box Hierarchy & Atom Forensics
*   **Atom Box Parsing & Layout Integrity:** Inspecting the hierarchical box topology of ISO Base Media File Format (ISOBMFF / ISO/IEC 14496-12) files (`ftyp`, `moov`, `mvhd`, `trak`, `mdia`, `minf`, `stbl`, `moof`, `mdat`).
*   **Box Ordering & Multiplexing Origins:** Genuine hardware camcorders and smartphones write video frames sequentially to disk in real-time, placing the massive `mdat` (media data) box before the index `moov` box, or using fragmented MP4 boxes (`moof`/`mdat` pairs). In contrast, desktop video editing suites (Adobe Premiere, DaVinci Resolve) and batch FFmpeg encoding pipelines commonly generate "fast-start" streaming MP4 files where the complete `moov` index is relocated to the very beginning of the container.
*   **Time-to-Sample & Chunk Offset Consistency:** Cross-checking `stts` (decoding time-to-sample), `ctts` (composition time-to-sample), `stss` (sync sample / IDR-frame table), and `stco` / `co64` (chunk offset tables). Asymmetric table offsets or corrupt delta counts reveal spliced frame ranges or truncated video streams.

### 20.2 Camera Hardware Telemetry, IMU, & GPS Streams
*   **Proprietary Sensor Telemetry Tracks:** Modern physical cameras embed dedicated metadata tracks synchronizing physical sensor telemetry to each video frame:
    *   *Sony Real-Time Telemetry (`rtmd`):* Frame-by-frame lens focal length, iris $f$-number, autofocus distance bracket, optical image stabilization (OIS) gyro vectors, and internal CMOS temperature.
    *   *GoPro Telemetry Format (GPMF):* Integrated 3-axis accelerometer ($200\text{ Hz}$), 3-axis rate gyro, GPS fix coordinates ($18\text{ Hz}$), camera temperature, and shutter exposure time per frame.
    *   *DJI Flight Telemetry Subtitles & Data Tracks:* Real-time barometric altitude, motor RPM, GPS coordinates, gimbal pitch/roll/yaw, and home-point distance embedded in subtitle tracks or private metadata tracks.
    *   *Apple QuickTime User Data & Timestamps:* Nanosecond-precise hardware monotonic timestamps (`com.apple.quicktime.creationdate`, `com.apple.quicktime.camera.exposure.mode`).
*   **Forensic Grounding:** Fully synthetic AI videos (Kling, Veo, Runway) generate bare video rasters lacking valid hardware telemetry streams; when telemetry is fraudulently synthesized or cloned from another video, physical discrepancies between camera motion vectors in optical flow and recorded IMU accelerometer trajectories expose the fraud.

### 20.3 Embedded Keyframe & Thumbnail Desynchronization
*   **Preview Poster / Thumbnail Mismatch:** MP4 and QuickTime containers often embed static JPEG poster frames in `moov/udta` or initial thumbnail sample tables. Video trimming and regional inpainting tools regularly alter the main video stream while leaving the original thumbnail untouched, preserving the uncensored or unedited physical scene within the container's own preview buffer.

### 20.4 Streaming Transport Container Integrity
*   **MPEG-TS PES Continuity Forensics:** MPEG Transport Streams (`.ts`) broadcast over DVB, ATSC, or UDP maintain a 4-bit continuity counter ($0\text{–}15$) per 13-bit Packet Identifier (PID). Discontinuities in continuity counters or jitter in Program Clock Reference (PCR) timestamps isolate video frame drops, broadcast splicing, or illegal packet injections.
*   **WebM / Matroska EBML Header Validation:** Analyzing Extensible Binary Meta Language (EBML) header tags in WebM streams. Generative video scripts often produce bare EBML clusters missing standard muxing application identifiers (`MuxingApp`, `WritingApp`) or exhibiting non-standard DocTypeVersion numbers.

### 20.5 Encoder Signatures & NAL Unit SEI Strings
*   **Supplemental Enhancement Information (SEI) NAL Units:** H.264/AVC, H.265/HEVC, and H.266/VVC bitstreams embed SEI messages. Open-source encoders (x264, x265) write explicit configuration strings into user-data SEI NAL units (`x264 - core 164 - r3108...`, encoding options, QP matrices). Hardware camera encoders (Sony BIONZ XR, Canon DIGIC, Apple A-series ISP) use proprietary SEI syntax without third-party open-source strings.
*   **Quantization Parameter (QP) Trajectories:** Extracting macroblock-level and CTU-level QP curves across time. Hardware cameras adapt QP smoothly based on scene illuminance and sensor gain; AI video post-processing or re-encoding introduces sharp, unnatural QP spikes across I-frame boundaries.

---

## 21. Cross-Cutting Framework: Legal, Safety, Cheapfakes, Granularity, & Security

Forensic signal verification operates within complex regulatory, human safety, and operational environments. Authenticity cannot be reduced solely to raw codec bitstreams or pixel flow; it intersects with law, narrative context, and security.

### 21.1 Trust, Safety, & Harm Classification Taxonomy
*   **Hard-Block Exploitation Classes:** Child Sexual Abuse Material (CSAM/CSAE) and Non-Consensual Intimate Imagery (NCII / deepfake pornography) represent absolute priority hard-block classes triggering immediate forensic isolation, cryptographic perceptual hashing (PhotoDNA, PDQ), and mandatory statutory reporting (e.g., NCMEC).
*   **Harm Vectors:** Severe violent extremism, self-harm instruction, illegal weapons/narcotics synthesis schematics, and coordinated targeted harassment.

### 21.2 Legal, Regulatory, & Evidentiary Framework
*   **Biometric Privacy Compliance:** Enforcement of GDPR Article 9 (biometric processing prohibitions without explicit consent), Illinois Biometric Information Privacy Act (BIPA, 740 ILCS 14/), Texas Capture or Use of Biometric Identifier Act (CUBI, Tex. Bus. & Com. Code § 503.001), and Washington State Biometric Privacy Law (RCW 19.375) regarding continuous facial geometry and gait tracking.
*   **EU AI Act Transparency Obligations:** Strict compliance with Regulation (EU) 2024/1689 Article 50, mandating that providers and deployers of AI systems generating or manipulating synthetic audio or video media mark the outputs in a machine-readable format and ensure prominent public disclosure.
*   **Intellectual Property & Licensing:** Identifying Public Domain (CC0), Creative Commons (CC-BY, CC-NC, CC-SA), broadcast sports licensing, and copyright infringement scraping signatures.
*   **Export Control Restrictions:** US International Traffic in Arms Regulations (ITAR, 22 CFR 120-130) and Export Administration Regulations (EAR, 15 CFR 730-774) restricting unauthorized processing and dissemination of satellite and aerial video with Ground Sample Distance (GSD) $< 0.3\text{ m}$.
*   **Medical Data Privacy (HIPAA / GDPR / EHDS):** De-identification compliance verifying that clinical endoscopic, laparoscopic, and surgical video recordings strictly satisfy HIPAA Safe Harbor (45 CFR § 164.514) 18 identifiers, GDPR Article 9 health exceptions, and European Health Data Space (EHDS) guidelines, ensuring zero leaked Protected Health Information (PHI) in telemetry tracks or burnt-in pixel rasters.
*   **Judicial & Evidentiary Admissibility:** Adherence to Federal Rules of Evidence (FRE 901 authentication, FRE 902(13)/(14) certified self-authenticating digital records), Daubert/Frye scientific reliability standards, and ISO/IEC 27037:2012 digital evidence handling and chain-of-custody logging.

### 21.3 "Real but Misleading": Contextual Cheapfakes & Temporal Re-use
*   **Contextual Cheapfakes (Shallowfakes):** Authentic, untampered historical optical video republished with falsified geotags, incorrect dates, or deceitful narrative captions (e.g., portraying a 2018 military exercise as live 2026 combat).
*   **Deceptive Temporal Clipping:** Selectively cutting video footage immediately before or after a critical event to radically invert the perceived context or justification of an action.
*   **Speed Alteration (Temporal Distortion):** Slowing down authentic speech video (e.g., to 75% speed) with pitch correction to falsely convey intoxication or neurological impairment.
*   **Forensic Grounding:** Forensic signal detectors verify that the pixel physics are authentic; external knowledge graphs, reverse video search (perceptual video hashing), and timestamp correlation are required to detect contextual falsehoods.

### 21.4 Spatio-Temporal Granularity & Localized Provenance
*   **Dense Spatio-Temporal Tube Masking:** Video manipulations are rarely applied globally across all frames. The validation engine outputs a 4D spatio-temporal segmentation tensor:
    $$M(x, y, t) \in [0, 1]$$
    isolating the precise bounding box tube and frame interval $[t_{\text{start}}, t_{\text{end}}]$ where neural inpainting, face-swapping, or visual removal occurred, while certifying surrounding spatial regions and frames as authentic optical capture.
*   **Per-Speaker Audio-Visual Diarization:** Multi-speaker scenes require independent visual face-tracking tubes bound to diarized audio speech tracks to detect isolated voice or face impersonation within a panel discussion.

### 21.5 Media Lifecycle & Platform Transcoding Laundering
*   **The Transcoding Cascade:** Video undergoes severe generational compression: Camera RAW (ProRes/All-I at 400 Mbps) $\to$ NLE Render (H.264 at 50 Mbps) $\to$ Social Messaging App (downscaled 720p H.264 at 1.5 Mbps, stripped container atoms) $\to$ Social Media Feed (re-encoded to AV1/VVC with Film Grain Synthesis).
*   **Anti-Forensic Laundering vs. Benign Degradation:** Detectors must distinguish between benign platform re-quantization and deliberate laundering schemes designed to wash away PRNU sensor traces and optical flow micro-discontinuities.

### 21.6 Video File Security, Container Polyglots, & Adversarial Attacks
*   **Container Polyglots:** Crafting binary files that parse simultaneously as valid MP4/ISOBMFF video and executable PE/ELF or ZIP archives, hiding payloads inside ignored movie atoms (`free`, `skip`).
*   **Parser Buffer Exploits:** Exploiting vulnerabilities in video demuxers (FFmpeg, libavcodec, GStreamer) via corrupted atom header lengths (`moov`, `trak`, `mvhd`), integer overflows, or malformed NAL units.
*   **Visual Adversarial Perturbations & Multi-Modal Prompt Injections:** Embedding imperceptible adversarial noise or high-contrast typographic text into video frames designed to hijack downstream Vision-Language Models (VLMs) or automated content moderation bots into bypassing security policies.

### 21.7 Detector Validity, Demographic Bias, & Domain Shift
*   **Demographic Parity & Fitzpatrick Skin Types:** Auditing neural facial artifact and deepfake discriminators across Fitzpatrick phototypes I through VI, lighting conditions, and diverse age groups to ensure balanced false-positive and false-negative rates.
*   **Domain Shift & Motion Dynamics Generalization:** Evaluating detector stability against extreme camera motion, low-light sensor noise, high-action sports motion blur, and varied shutter angles ($90^\circ\text{ to }360^\circ$).
*   **Adversarial Robustness:** Benchmarking detector resilience against temporal perturbation attacks, frame-rate decimation, and anti-forensic temporal smoothing filters.

### 21.8 Accessibility Tracks & Multimodal Parity
*   **Accessibility Metadata Forensics:** Inspecting embedded EIA-608 / CEA-708 closed caption packets, WebVTT subtitle tracks, and SMPTE-TT / TTML XML streams.
*   **Multimodal Semantic Verification:** Detecting discrepancies where generative AI video streams are paired with recycled, contradictory human-authored caption tracks, or where subtitle text contradicts spoken dialog and visual actions.

### 21.9 Open-Set Source Attribution & Novel Generator Identification
*   **Open-Set Attribution Architecture:** Avoiding closed-world classification assumptions ($k \in \{1, \dots, K\}$ known model families). The engine implements an open-set attribution head that computes latent Mahalanobis distances against known generator cluster centers $\{\boldsymbol{\mu}_k\}$. When $\min_k D_{\text{attr}}(\mathbf{z}, \boldsymbol{\mu}_k) > \tau_{\text{attr}}$, the system outputs `UNKNOWN_SOURCE / NOVEL_GENERATOR`, triggering signature clustering and automated quarantine.

---

## 22. Forensic Physics & Mathematical Differentiation Formulations

| Forensic Indicator | Authentic Optical Video | Conventional Edit | AI-Enhanced Hybrid | Fully AI-Generated | Screen Recording | Neuromorphic DVS |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Temporal PRNU ($\rho_{\text{temp}}$)\*** | $\rho \ge 0.15, \text{PCE} \ge 50$ | Breaks at edit cuts | Disrupted in manipulated patch | $\rho \approx 0, \text{PCE} < 10$ | $\rho < 0.02, \text{PCE} < 10$ | **N/A** (Event stream) |
| **Optical Flow Error (OFCE)\*** | $\text{OFCE} < 0.5\text{ px}$ | Spikes at cuts | Localized discontinuity | Non-Newtonian flow anomalies | Discrete UI jumps | Continuous event flow |
| **Rolling Shutter Skew\*\*** | Present on CMOS sensors | Preserved | Inconsistent in swapped face | **Absent** (Instant rendering) | **Absent** (Framebuffer) | Zero rolling shutter |
| **ZKP / C2PA Verification** | Hardware signed | zk-SNARK proof | C2PA composite assertion | C2PA synthetic tag | Software recorder tag | Asynchronous stream signature |

*\*Note: Temporal PRNU correlation $\rho \ge 0.15$ and OFCE $< 0.5\text{ px}$ are empirical operational thresholds calibrated against standardized benchmark corpora (e.g., VISION dataset, Dresden Image Database); actual operational cutoffs adapt dynamically to frame bitrate, resolution, and sensor gain.*  
*\*\*Critical Note on Physical Indicators: The absence of a physical marker does NOT determinatively prove synthetic generation. Global-shutter CMOS sensors (e.g., Red Komodo, Sony Pregius) exhibit zero rolling-shutter skew; advanced temporal noise reduction (TNR) and computational HDR video fusion heavily suppress sensor noise without generative fabrication; and battery-powered recordings exhibit zero Electric Network Frequency (ENF).*

---

## 23. System Architectural Integration & Calibrated Bayesian Verification

The operational taxonomy defines the target `project-content-validation` pipeline using a decoupled, two-axis decision architecture. Stage names below are logical stages of the target architecture; not every stage maps to an implemented module in the current codebase (e.g., `optical` and `crypto` are specification-level stages).

```
[ Input Video / Event Stream ] ───> [ 1. Frame / Event Demuxing ]  ───> Keyframe sampling, DVS AER decoding
                               ───> [ 2. video_detector.temporal ]  ───> Frame-to-frame PRNU, Optical Flow (OFCE)
                               ───> [ 3. video_detector.optical ]   ───> Rolling shutter, Stokes polarimetry
                               ───> [ 4. video_detector.codecs ]    ───> GOP structure, motion vector stego
                               ───> [ 5. video_detector.audio ]     ───> Phoneme-viseme lip-sync, room acoustics
                               ───> [ 6. video_detector.crypto ]    ───> C2PA video manifests, zk-SNARK proofs
                               ───> [ 7. Deep Video Transformer ]   ───> Spatio-temporal neural representation
                                                     │
                                                     ▼
                              [ Epistemic Out-of-Distribution Check ]
                                   D_OOD = D_Mahalanobis(z, Z_in-dist)
                                                     │
                          ┌──────────────────────────┴──────────────────────────┐
                          ▼                                                     ▼
               D_OOD > τ_OOD (Out-of-Distribution)                   D_OOD ≤ τ_OOD (In-Distribution)
                          │                                                     │
                          ▼                                                     ▼
            [ OUT-OF-DISTRIBUTION (OOD) /                 [ Calibrated Bayesian LLR Fusion ] (scoring.py)
              UNSEEN GENERATOR QUEUE ]                         L = Σ w_i ln(P(x_i|AI) / P(x_i|Real))
            - Overrides nominal probability P                   P(AI | x) = 1 / (1 + e^-L)
            - Triggers forensic human review                                    │
            - Preserves open-world safety                 ┌─────────────────────┼─────────────────────┐
                                                          ▼                     ▼                     ▼
                                                    P(AI) ≥ 0.995        0.400 ≤ P ≤ 0.600      P(AI) ≤ 0.005
                                                     (High AI)            (Inconclusive)         (High Real)
```

### Calibrated Probability Bands
As governed by the logistic formulation $P = \frac{1}{1 + e^{-\mathcal{L}_{\text{video}}}}$, real-world probabilities asymptote toward but never strictly reach $1.0$ ($100\%$) or $0.0$. Decision boundaries enforce five distinct operational bands:

1.  **High-Confidence Synthetic / AI-Generated ($P(\text{AI}) \ge 0.995$):** Unambiguous generative synthesis artifacts, broken temporal PRNU, or synthetic C2PA assertions.
2.  **Leaning Synthetic / Flagged for Review ($0.600 < P(\text{AI}) < 0.995$):** Notable spatio-temporal anomalies or partial generative inpainting detected; automated platform warning applied.
3.  **Inconclusive / Indeterminate Evidence ($0.400 \le P(\text{AI}) \le 0.600$):** Equivocal signals resulting from heavy compression, format conversion, or conflicting multi-feature evidence; flagged for forensic specialist review.
4.  **Leaning Authentic / Low Anomaly ($0.005 < P(\text{AI}) < 0.400$):** Minor non-generative post-processing or standard compression artifacts consistent with physical capture.
5.  **High-Confidence Authentic Physical Capture ($P(\text{AI}) \le 0.005 \iff P(\text{Real}) \ge 0.995$):** Robust physical sensor noise, continuous temporal PRNU, valid optical kinematics, intact hardware telemetry, and verified cryptographic provenance.

> **Orthogonal Channels:** The probability bands apply only to the in-distribution authenticity score. Three channels operate independently of it: (a) the Trust & Safety pre-gate (CSAM/NCII hard-blocks) short-circuits all bands; (b) a valid ZK-verified or hardware-signed edit proof is reported as an `AUTHENTIC_EDITED_VIDEO` provenance label alongside, not inside, the band; (c) contextual/cheapfake claims are assessed separately from signal authenticity (Section 21.3).

---

## 24. Conclusion & Strategic Roadmap

1.  **Systematic Multi-Dimensional Grounding:** By unifying temporal sensor physics, optical flow kinematics, rolling shutter timings, multi-modal audio-visual synchronization, video container architectures, and deep neural latent representations, the taxonomy establishes a robust, mathematically defensible framework across all operational video modalities.
2.  **Humility in Open-World Forensics:** Recognizing that video synthesis and generative world modeling constitute open classes with rapidly evolving architectures and counter-forensic attacks, the system explicitly decouples epistemic uncertainty via `OUT-OF-DISTRIBUTION (OOD)` gating and provides an open-set `UNKNOWN_SOURCE` attribution state.
3.  **Multi-Evidence Fusion is Essential:** Physical sensor validation, cryptographic assertions, and statistical learning reinforce one another, preventing single-point failure modes across the content lifecycle.

---

## 25. References & Citations

1.  **C2PA:** *Technical Specification for Digital Content Provenance, Version 2.4 (2026).* `https://c2pa.org/`
2.  **INRS Ultrafast Optical Laboratory:** *SCARF: Swept-Coded Aperture Real-Time Femtophotography at 156.3 Trillion Frames Per Second.*
3.  **Event Horizon Telescope Collaboration:** *Dynamic Spatio-Temporal Imaging of Sagittarius A\* Accretion Flows.*
4.  **Prophesee & Sony:** *Stacked Event-Based Metavision Sensor (IMX636) Technical Architecture.*
5.  **Kuaishou Technology:** *Kling 3.0 Architecture: Spatio-Temporal Flow Matching Video Synthesis.*
6.  **SMPTE:** *SMPTE ST 2110: Professional Media Over Managed IP Networks.*
7.  **VISION Dataset:** Shullani, D., et al. (2017). *VISION: A Video and Image Dataset for Source Identification.*
8.  **Dresden Image Database:** Gloe, T., & Böhme, R. (2010). *The 'Dresden Image Database' for Benchmarking Digital Image Forensics.*
9.  **FaceForensics++:** Rössler, A., et al. (2019). *FaceForensics++: Learning to Detect Manipulated Facial Images.*
10. **ISO/IEC 27037:2012:** *Information technology — Security techniques — Guidelines for identification, collection, acquisition and preservation of digital evidence.*
11. **W3C:** *Media Source Extensions (MSE) & WebRTC 1.0 Specifications.*

---

## Appendix A: Controlled Vocabulary Standards Mapping

| Standard | Identifier | Human-Readable Label | Engine State Mapping |
| :--- | :--- | :--- | :--- |
| **IPTC** | `digitalsourcetype:digitalCapture` | Original Digital Video Capture | `AUTHENTIC_REAL_VIDEO` |
| **IPTC** | `digitalsourcetype:screenCapture` | Screen Recording / Framebuffer | `AUTHENTIC_SCREEN_RECORDING` |
| **IPTC** | `digitalsourcetype:trainedAlgorithmicMedia` | Created using Generative Video AI | `FULLY_AI_GENERATED_VIDEO` |
| **C2PA** | `c2pa.actions.zk_video_trim` | ZK-Proven Video Trimming | `AUTHENTIC_EDITED_VIDEO` (ZK-Verified) |
| **AER**  | `dvs.event_stream` | Neuromorphic Event Cloud | `AUTHENTIC_NEUROMORPHIC_STREAM` |
| **Engine** | OOD gate ($D_{\text{OOD}} > \tau_{\text{OOD}}$) | Out-of-Distribution / Unseen Generator | `OOD_UNSEEN_GENERATOR` |
| **Engine** | Probability band 3 ($0.400 \le P(\text{AI}) \le 0.600$) | Inconclusive / Indeterminate Evidence | `INCONCLUSIVE` |
| **Engine** | Open-set attribution ($\min_k D_{\text{attr}} > \tau_{\text{attr}}$) | Unknown Source / Novel Generator | `UNKNOWN_SOURCE` |
| **Engine** | Trust & Safety pre-gate (CSAM / NCII) | Hard-Block Class (bypasses probability bands) | `HARD_BLOCK_ESCALATE` |

---

## Appendix B: Canonical Generative AI Video Model Specifications

```
Model                  Max Resolution    Max FPS    Max Duration    Architecture
───────────────────────────────────────────────────────────────────────────────────────────────
Kling 3.0 / Omni       4K (3840×2160)    60 fps     120s+ (Chained) VideoDiT + Flow Matching
Google Veo 3.1         Native 4K         30 fps     60s+            VideoDiT + Flow Matching
Runway Gen-4.5         4K (Upscaled)     30 fps     60s+            VideoDiT + Motion Brush
Seedance 2.5           4K                30 fps     120s+ (Chained) VideoDiT + Reference Chain
```

---

## Appendix C: Video Codec Forensic Signature & Compression Matrix

| Codec | Block Structure | Inter-Frame Prediction | Forensic Tells |
| :--- | :--- | :--- | :--- |
| **H.264/AVC** | $16\times 16$ macroblocks | P/B frames, 1/4-pel motion estimation | Visible $16\times 16$ grid at low bitrates; deblocking smoothing bands. |
| **H.265/HEVC** | CTUs up to $64\times 64$ | Merge mode, directional intra prediction | Larger block boundaries; complex prediction residual patterns. |
| **H.266/VVC** | Multi-type tree ($128\times 128$) | Affine motion, adaptive loop filtering | Non-grid structural patterns in smooth areas; highly flexible blocks. |
| **AV1** | $128\times 128$ superblocks | Film Grain Synthesis (FGS) | Synthetic grain post-decode; temporal smearing at low bitrates. |

---

## Appendix D: Legacy & Historical Video Format Timeline

```
Year    Format              Type                    Resolution Equivalent
────────────────────────────────────────────────────────────────────────────────
1956    2-inch Quadruplex   Transverse scan tape    ~350 lines (analog)
1969    U-matic             Helical scan 3/4" tape  ~250 lines (analog)
1976    VHS                 Consumer 1/2" tape      ~240 lines (analog)
1986    D-1                 Digital uncompressed    720×486 / 720×576
1995    DV25 (MiniDV)       Digital 1/4" tape       720×480 / 720×576
2003    H.264/AVC           Digital codec standard  Up to 8K
2013    H.265/HEVC          Digital codec standard  Up to 8K
2020    H.266/VVC           Digital codec standard  Up to 16K
2026    VideoDiT Flow       Generative Video Stream Native 4K 60fps
```

---

## Appendix E: Mathematical Physics Formulations for Temporal Forensics

### 1. Frame-to-Frame Temporal PRNU Correlation & PCE
$$\rho_{\text{temp}}(t, t+1) = \frac{\sum_{x,y} W_t(x,y) \cdot W_{t+1}(x,y)}{\sqrt{\sum_{x,y} W_t^2(x,y)} \cdot \sqrt{\sum_{x,y} W_{t+1}^2(x,y)}} \ge \tau_{\text{PRNU}} \approx 0.15, \quad \text{PCE} \ge 50$$
*(Note: $\tau_{\text{PRNU}} \approx 0.15$ represents an empirical threshold calibrated against VISION dataset and Dresden Image Database baseline sequences; subject to frame bitrate, resolution, and sensor gain).*

### 2. Optical Flow Consistency Error (OFCE)
$$\text{OFCE}(t) = \frac{1}{|P|} \sum_{(x,y) \in P} \| F_{t \to t+1}(x,y) + F_{t+1 \to t}\big(x + F_{t \to t+1}(x,y)\big) \|_2 < 0.5\text{ px}$$
*(Note: $\text{OFCE} < 0.5\text{ px}$ represents empirical baseline consistency on natural cinematic motion vectors).*

### 3. Spatio-Temporal Structural Consistency (STSC)
$$\text{STSC} = \frac{1}{T} \sum_{t=1}^{T-2} \|\mathbf{x}(t+2) - 2\mathbf{x}(t+1) + \mathbf{x}(t)\|_2$$

### 4. Bayesian Spatio-Temporal Evidence Pooling
$$\mathcal{L}_{\text{video}} = \sum_{i=1}^{N_{\text{spatial}}} w_i \ln \left( \frac{P(x_i \mid \text{AI})}{P(x_i \mid \text{Real})} \right) + \sum_{j=1}^{N_{\text{temporal}}} w_j \ln \left( \frac{P(x_j \mid \text{AI})}{P(x_j \mid \text{Real})} \right), \quad P(\text{AI} \mid \mathbf{x}) = \frac{1}{1 + e^{-\mathcal{L}_{\text{video}}}}$$


---

## Appendix F: Implementation Status (REV7)

Status key: **Implemented** (code + tests), **Partial** (some sub-items), **Recognition-only** (format sniffed, no authenticity scoring), **Spec-only** (documented taxonomy, no code). "Advisory" findings never change P(AI). The shared foundation (`core/forensics/`, `core/bands.py`, `ui/stages.py`) is now wired into the **image**, **audio** and **video** modalities. Code references are relative to `video_detector/` unless stated.

| Dimension / Section | Status | Where | Notes |
| :--- | :--- | :--- | :--- |
| **A** Provenance & authenticity spectrum | Partial | `scoring.py`, `detector.py` | Temporal heuristics only. Virtual-camera injection (State 11) cannot be recovered from a file; only OBS/Streamlabs/ManyCam-style capture-software strings are reported (`dimension_checks/container.py`). |
| **B** Genre & subject | Partial | `content.py` | |
| **C** Art styles / AR filters | Spec-only | - | |
| **D** EM spectrum, radar/sonar, depth video | Spec-only | - | |
| **E-J** Astrophysical, ultrafast, polarization, remote sensing, event streams, BCI | Recognition-only / Spec-only | `core/forensics/gates.py` | DICOM (cine), HDF5, NetCDF and AEDAT event-stream files are recognized and not scored; no analysis of these modalities. |
| **K** Temporal dynamics | Partial | `temporal.py`, `dimension_checks/signal.py` | Inter-frame motion variance and flicker (existing); row-combing interlace detection with container `fiel` flag; duplicate-frame cadence (3:2 pulldown, regular duplication, heavy duplication, isolated duplicates, hard cuts). Rolling-shutter and optical-flow-consistency analysis: spec-only. |
| **L** Spatial geometry / 3D / VR | Spec-only | - | |
| **M** Codecs, streaming, telemetry, containers | Partial | `dimension_checks/container.py` | MP4/MOV box layout (brands, fast-start, fragmentation), sample-table consistency (`stts`/`stsz`/`stss`/`stco`/`stsc`, CFR/VFR), `mvhd` time plausibility, tool strings, telemetry-track presence (GoPro `gpmd`, Sony `rtmd`, Google `camm`, DJI), x264/x265/libavcodec bitstream strings, MKV/WebM EBML writer strings. HLS/DASH segment forensics, WebRTC loss artifacts and telemetry-vs-optical-flow validation: spec-only. |
| **N** Audio-visual synchronization | Partial | `cross_modal.py` | Modality-asymmetry and scene-plausibility checks against an externally supplied audio result; no phoneme-viseme analysis. |
| **O** Generative AI & world models | Partial | `attribution.py` | 15-generator attribution (8 calibrated). Explicit generator names in container software/comment fields are scored (+0.40). |
| **P** Watermarking, crypto, security, stego | Partial | `provenance.py`, `dimension_checks/integrity.py` | C2PA is a byte-signature presence scan plus detection of the C2PA UUID box, **not** cryptographic validation. Implemented: format sniff, bytes beyond the container end (incl. truncation), polyglot signatures, instruction-like tag text. Watermark detectors and steganalysis: spec-only. |
| **Sec. 20** Metadata & container forensics | Implemented (weak evidence) | `dimension_checks/container.py` | See Dimension M. SEI NAL decoding beyond text strings, QP trajectories: spec-only. |
| **Sec. 21.1** Harm / hard-block | Partial | `core/forensics/gates.py` | Pluggable SHA-256 hard-block list (no classifier). Violence/hate classification: spec-only. |
| **Sec. 21.2** Legal flags | Implemented (advisory) | `dimension_checks/legal.py` | Rights notice, embedded location / GPS telemetry, biometric notice (faces), AI-disclosure label. Export-control and PHI scans: spec-only. |
| **Sec. 21.3** Cheapfakes / context | Partial | `dimension_checks/context.py` | 16-frame perceptual-hash fingerprint + optional local reference index; no external reverse search. |
| **Sec. 21.4** Granularity | Partial | `temporal.py` | Temporal segments; no spatio-temporal tube masks. |
| **Sec. 21.5** Lifecycle laundering | Implemented (heuristic) | `dimension_checks/lifecycle.py` | Re-encoding likelihood (bits per pixel, platform sizes, web-export layout, stripped metadata). |
| **Sec. 21.6** File security | Implemented | `dimension_checks/integrity.py` | See Dimension P. Demuxer-exploit detection: spec-only. |
| **Sec. 21.7** Detector validity | Partial | `dimension_checks/reliability.py` | Confidence limiters (resolution, short clips, compression, interlacing, duplicated cadence, VFR, re-encoding). Demographic-parity audits and domain-shift benchmarks: spec-only. |
| **Sec. 21.8** Accessibility | Spec-only | - | Caption-track forensics not implemented. |
| **Sec. 21.9** Open-set attribution | Implemented (basic) | `dimension_checks/__init__.py` | `UNKNOWN_SOURCE` when no generator profile matches an AI-leaning video. |
| **Sec. 22** Five probability bands | Implemented | `core/bands.py` | Applied to the in-distribution score. |
| **Sec. 22** OOD gate | Implemented, uncalibrated by default | `core/forensics/ood.py`, `dimension_checks/fit_ood.py` | 5-dim temporal feature vector; reports `NOT_CALIBRATED` until fitted (`python -m video_detector.dimension_checks.fit_ood`). |

**Hybrid scoring (as built):** only `PHYSICAL_SIGNAL` / `METADATA_WEAK` findings may add log-odds (base-10, positive = toward AI): per-finding cap 0.25 (explicit generator string 0.40), total clamp +/-0.40, absence never scored. Video pools by weighted average, so the terms shift the pooled probability in log-odds space (`VideoAIDetector.analyze_video(extra_log_lrs=...)`); with no terms the result is unchanged. The only scoring term is an explicit generative-tool name in container software/comment fields (+0.40). Interlacing and cadence findings are informational and carry no log-odds.

**Known limitation (pre-existing):** the Stage 5 narrative text in the Streamlit video flow can show 0 x 0 pixels / 0.0 s / 0.0 fps because it reads profile keys the video profiler does not populate under those names; this predates the dimension checks and is not changed here.


**Post-audit corrections (2026-10-03), all covered by tests:**
- Scores are heuristic and uncalibrated (`calibration_status`); thresholds were tuned on synthetic fixtures only. `python -m services.calibration_cli --modality video` measures accuracy, ECE and band occupancy on the held-out validation split once a labeled media library exists.
- C2PA is marker presence only (no signature validation): reported, never scored, never called "verified".
- Camera EXIF is unauthenticated: coherent EXIF earns no credit; EXIF contradicted by strong synthetic pixel evidence is demoted (image detector trust policy).
- The package pipelines and the Streamlit flow share one verdict path (`core.decision.generate_final_decision`, `decision_mode` = `image_authoritative` | `fused`); attribution is explanation only and is not double counted.
- Parser fixes: RIFF chunk walking is seek-based (no false WARN beyond 4 MB), PNG text chunks after IDAT are found.

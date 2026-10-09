# Dataset specification: what to collect to train and test a Likely AI / Likely Real / Unsure detector

Written October 2026 from public sources and the model landscape as of mid-2026. Generator and phone names change monthly:
treat the lists as a starting point and re-check them every quarter. Counts are targets per row; "test" images are never
used for training or threshold tuning.

## Rules that matter more than the counts

1. **Label certainty.** A "real" image must be provably real: your own captures, camera-RAW research sets, or datasets
   published before 2022. An "AI" image must have a known generator and, ideally, the prompt and settings.
2. **Same treatment for both classes.** Real and AI images must share the same size, format and compression ranges, or the
   model learns those instead of AI-ness (the first face model did exactly that).
3. **Generators held out.** Keep at least 10 generators (and 10 camera models) entirely out of training. The score on those is
   the honest accuracy.
4. **Every row exists in two states:** the original file, and the file after the real-world journey (WhatsApp, Instagram,
   screenshot, re-save). Most images you will be asked about have been through at least one of them.
5. **Record metadata for every image** (source, device or generator, date, edits applied, transformation chain), but never
   copy personal photos into the repository; reference them by path and content hash.

## A. Real images

### A1. Phones (the bulk of real images). Test: 300 per row; train: 2,000 per row where available

| Group | Devices to cover (2024-2026 models plus older ones still in use) |
|---|---|
| Apple | iPhone 17 / 17 Pro / Pro Max / Air, 16 series, 15 series, 13/12 (still common), iPhone SE |
| Samsung flagship | Galaxy S26 Ultra / S26, S25 series, S24 series, Z Fold / Flip |
| Samsung volume | Galaxy A series (A15, A25, A35, A55, A06), M series |
| Google | Pixel 10 / 10 Pro XL, Pixel 9, Pixel 8, Pixel 7a |
| Xiaomi family | Xiaomi 17 Ultra / 15 / 14, Redmi Note 14 / 13, Redmi 13C, POCO X7 / F7 |
| Oppo, Vivo, OnePlus, Realme | Find X9, Vivo X300, X200, Vivo Y series, OnePlus 13 / Nord, Realme 14 / GT |
| Huawei, Honor | Pura 80, Mate 70, Honor Magic 7 / 8, Honor X series |
| Others | Motorola (Edge, G series), Nothing Phone, Sony Xperia, Asus ROG/Zenfone, Fairphone |
| Emerging-market volume | Infinix, Tecno, Itel, Lava, Micromax, Jio phones |
| Older phones | 2012-2020 Android and iPhone |

For each phone, include: main camera, ultrawide, telephoto/zoom (including AI-assisted zoom of 10x or more), front/selfie
camera, portrait mode, night mode, HDR, panorama, Live/Motion photo stills, beauty mode on and off, and RAW/ProRAW plus the
processed JPEG/HEIC pair.

### A2. Dedicated cameras. Test: 300 per row; train: 1,500

Canon (EOS R series, DSLR), Nikon (Z series, D series), Sony (alpha, RX, and old Cyber-shot DSC compacts),
Fujifilm (X, GFX), Panasonic Lumix, OM System / Olympus, Leica, Pentax, GoPro and action cams, DJI drones and Osmo,
360 cameras, webcams and laptop cameras, CCTV and dashcam frames, scanners and film scans, medical/microscope (only if you
need them).

### A3. Where real images come from, as a dataset

COCO, Open Images, ImageNet (pre-2022), Flickr (CC-licensed, filter by upload date before 2022), Unsplash/Pexels (check for AI
uploads after 2023), RAISE and MIT-Adobe FiveK (RAW pairs), Dresden Image Database and VISION (camera-model fingerprints),
FFHQ and CelebA for faces, plus your own captures.

### A4. Real content types (every device row should mix these). Target 15 types

Portraits and selfies (all skin tones, ages, glasses, masks), groups and crowds, full-body, children, street scenes, indoor
rooms, food, products, documents and receipts photographed, whiteboards and screens (recaptured), landscapes, night and
low-light, sports/action, animals and pets, vehicles, architecture and interiors, events (weddings, festivals, temples,
classrooms).

### A5. Journeys (apply to a subset of every real row). Target 10 transformations

Original; WhatsApp (image and "document"); Instagram; Facebook; X/Twitter; Telegram; Snapchat; screenshot of the image;
re-photograph of a screen; JPEG re-save at quality 30-95; resize 25-200 %; crop; HEIC/AVIF/WebP conversion; email/cloud
re-encode.

### A6. Real but edited (still "Likely Real"). Target 2,000 train / 500 test per tool family

Lightroom, Snapseed, Photoshop (non-generative), GIMP, Canva, VSCO, Instagram filters, phone gallery editors, crop/rotate,
colour grading, sharpening/denoising (non-generative), background removal (classical), stickers/text overlays, HDR merges,
beauty apps in mild mode (Meitu, B612, Snow, YouCam).

## B. AI images

### B1. Closed / hosted generators (test 500, train 2,000 per row; as many as you can access)

| Family | Generators to cover |
|---|---|
| OpenAI | GPT Image 2 (and gpt-image-1.5/1), DALL·E 3, ChatGPT/Sora image output |
| Google | Nano Banana 2 (Gemini 3.1 Flash Image), Nano Banana Pro, Imagen 4, Imagen 3, Whisk/ImageFX, Gemini in Photos |
| Midjourney | V8.1, V8, V7, V6.1, niji 6/7 |
| Adobe | Firefly Image 4/5, Photoshop Generative Fill/Expand/Remove |
| Other Western | Ideogram 3, Recraft V3/V4, Leonardo (Phoenix, Lucid), Krea, Reve, Luma Photon, Runway, Canva AI, Microsoft Designer / MAI-Image, Meta AI (incl. WhatsApp/Instagram), Grok Imagine (xAI), Stability API, Playground |
| Chinese | Seedream (ByteDance/Doubao/Jimeng), Kling and Kolors, Hunyuan Image 3.0, Qwen-Image / Wan stills, Hailuo/MiniMax, Vidu, Baidu ERNIE-ViLG |

### B2. Open-weight generators (test 500, train 2,000 per row)

FLUX.2 (dev, klein), FLUX.1 (dev, schnell, Kontext), Stable Diffusion 1.5, 2.1, XL, 3, 3.5, Stable Cascade, Sana, PixArt-alpha/sigma,
HiDream, Lumina, Qwen-Image (+Edit), HunyuanImage 3, CogView4, OmniGen2, Kandinsky 5, Playground v2.5/v3, DeepFloyd IF, Wuerstchen,
Chroma, and community fine-tunes and LoRAs (Pony, Illustrious, RealVisXL, Juggernaut, epiCRealism). Community Forensics shows that
**number of distinct generators matters more than images per generator**: sample at least 50 different checkpoints from Civitai /
Hugging Face with 100-200 images each.

### B3. Older and other synthetic families (test 300, train 1,000 per row)

StyleGAN 1/2/3 (thispersondoesnotexist and similar), BigGAN, ProGAN, GigaGAN, VQGAN/DALL·E 1/Mini, Latent Diffusion, GLIDE,
autoregressive models (Parti, Janus, Emu3, Show-o). These are the families public benchmarks already contain.

### B4. AI-modified real images (the hardest class; counts per method, test 500 / train 2,000)

| Kind | Tools |
|---|---|
| Generative edits | Photoshop Generative Fill/Expand/Remove, Samsung Generative Edit and Sketch-to-image, Pixel Magic Editor / Reimagine / Add Me, Apple Clean Up, Google Photos Magic Eraser/Editor, Snapchat/Instagram AI edits, Meta AI edits |
| Inpainting / outpainting | SD/SDXL/FLUX inpainting, Kontext, Qwen-Image-Edit, GPT-Image edits, Nano Banana edits, Midjourney editor/retexture |
| Super-resolution / restoration | Topaz Photo AI, Magnific, Real-ESRGAN, Upscayl, CodeFormer, GFPGAN, Remini, Lightroom Enhance/Denoise, phone "AI zoom" |
| Face swap and reenactment | inswapper / InsightFace, FaceFusion, Roop, DeepFaceLab, SimSwap, Reface, Snapchat lenses, talking-head frame grabs |
| Style and colour | Style transfer, colourisation (DeOldify), AI beautification in strong mode, AI relighting (IC-Light), background replacement with generation |
| Video-model stills | Frames from Sora 2, Veo 3.x, Kling 2.x, Runway Gen-4, Hailuo, Wan 2.x, Seedance |

### B5. AI content types (every generator row should mix these). Target 20 types

Photoreal portraits (diverse ethnicity, age, expression), selfies, groups and crowds, full-body with hands, historical-style
photos, news/event-style photos, street and travel scenes, indoor interiors, landscapes, architecture, food, products and
ads, animals and wildlife, vehicles, text-heavy posters/menus/signs, fake screenshots and UI, fake documents (IDs, receipts,
certificates), memes, illustrations/anime/3D renders/paintings, satellite/aerial, low-resolution thumbnails and avatars.

### B6. Same journeys as A5

Every AI row must also exist after WhatsApp, Instagram, screenshot, re-save, resize, crop and re-photograph. Add adversarial
variants: added noise or film grain, blur, "humanizer" tools, EXIF forging, upscaling of a real-sized image.

## C. Targets in numbers

| Set | Real | AI (incl. AI-modified) | Notes |
|---|---|---|---|
| Training, minimum useful | 40,000 | 40,000 | about 30 devices, 40 generators, 5 journeys; public datasets can supply most |
| Training, recommended | 150,000 | 150,000 | 60+ devices, 100+ generators/checkpoints, all journeys |
| Held-out test (never trained on) | 15,000 | 15,000 | 10 generators and 10 camera models absent from training |
| "Hard" set (grows over time) | 2,000 | 2,000 | every error found in the field, with its true label |
| Abstention set | 1,000 | 1,000 | very low resolution, heavy compression, screenshots, tiny thumbnails: correct answer is Unsure |

Public datasets that already cover much of this (check licences and download terms before use): Community Forensics (2.7 M images,
4,803 generators), GenImage, WildFake (about 2.6 M AI / 1.0 M real), Chameleon (11 K AI / 15 K real), NTIRE 2026 robust-detection set
(108,750 real / 185,750 AI, 42 generators, 36 transformations), AI-GenBench, DailyBench, FakeBench.

## D. What not to collect

- Images of people who have not consented to be in a training set, beyond what the public datasets already include.
- Scraped images with unclear licences or uploaded after 2022 and labelled "real" without proof.
- Wallpapers, stock art and illustrations labelled real: their label is ambiguous.

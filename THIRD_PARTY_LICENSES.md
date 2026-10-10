# Third-party licences

This project's own code is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). Everything below is somebody else's work and stays
under its own licence. "Verified" means the licence was read from the upstream model card or licence file on 2026-10-10; nothing here is
legal advice. If you plan commercial use, read the rows marked **caveat** first.

## Models and weights

| Model | Used for | Licence | Where it lives | Verified from |
|---|---|---|---|---|
| YuNet (`face_detection_yunet_2023mar.onnx`) | face detection | MIT, Copyright (c) 2020 Shiqi Yu | `core/models/` (in the repository) | OpenCV Zoo `LICENSE` file |
| SFace (`face_recognition_sface_2021dec.onnx`) | same-person matching | Apache-2.0 | `core/models/` | OpenCV Zoo `LICENSE` file |
| MobileFaceNet expression model (`facial_expression_recognition_mobilefacenet_2022july.onnx`) | facial expression | Apache-2.0 (the OpenCV Zoo directory states it covers all its files) | `core/models/` | OpenCV Zoo directory page |
| MiVOLO v2 (`iitolstykh/mivolo_v2`, 28.8 M parameters) | apparent age, minor screening | Apache-2.0 (weights and network code) | `core/models/mivolo_v2/` and `core/perception/mivolo_vendor/` (licence file included) | Hugging Face model card |
| RF-DETR Small (`Roboflow/rf-detr-small`) | people, animals, vehicles, objects | Apache-2.0 | downloaded to the Hugging Face cache by `python -m services.fetch_models` | model card |
| SigLIP 2 Base (`google/siglip2-base-patch16-224`) | scene, species, vehicle type, genre, age-group opinion | Apache-2.0 | downloaded to the Hugging Face cache | model card |
| ResNet-18 ImageNet-1K starting weights (torchvision) | initial weights for the face-authenticity model and for any image or video backbone you train | **caveat:** torchvision's documentation states no licence for the weights; they were trained on ImageNet-1K, whose terms of access allow non-commercial research and education | downloaded by torchvision when training starts | torchvision documentation |
| `face_authenticity.pt` (trained by this project) | real versus AI-generated face | the project's licence for the code; **caveat:** it was fine-tuned from the ImageNet weights above on the Kaggle "Human Faces Dataset" (kaustubhdhote), whose licence could not be confirmed (mirrors list CC BY-SA 4.0 and "Other") | `image_detector/models/` (Git LFS) | training metadata inside the checkpoint |

Not shipped, optional, and trained by you if you want them: the image, video and audio AI-versus-real backbones and the out-of-distribution
gates. They carry the licence of whatever data you train them on.

## Datasets used only to evaluate (never redistributed)

UTKFace (non-commercial research), LAGENDA (CC BY 2.0), FairFace (CC BY 4.0): see `docs/AGE_DATASETS.md`.

## Python packages (versions installed when this was written, Python 3.14)

| Package | Version | Licence (from the installed package metadata) |
|---|---|---|
| streamlit | 1.65.0 | Apache-2.0 |
| numpy | 2.5.3 | BSD-3-Clause and others (0BSD, MIT, Zlib, CC0-1.0 for bundled parts) |
| pandas | 3.0.6 | BSD-3-Clause |
| torch | 2.14.1 | BSD-style (the wheel lists Apache-2.0 and BSD parts) |
| torchvision | 0.29.1 | BSD |
| opencv-python | 5.0.0.93 | Apache-2.0 (the wheels also bundle FFmpeg libraries under the LGPL; see the opencv-python project) |
| pillow | 12.3.0 | MIT-CMU (HPND) |
| requests | 2.34.2 | Apache-2.0 |
| transformers | 5.19.0 | Apache-2.0 |
| timm | 1.0.30 | Apache-2.0 |
| safetensors | 0.8.0 | Apache-2.0 |
| huggingface-hub | 2.2.0 | Apache-2.0 |
| pytest (development) | 9.1.1 | MIT |

`ffmpeg` is an external program you install yourself; it is not bundled and keeps its own (LGPL or GPL, depending on the build) licence.

## YuNet licence text (MIT)

```
MIT License

Copyright (c) 2020 Shiqi Yu <shiqi.yu@gmail.com>

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge,
publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH
THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
```

The Apache-2.0 text is in `LICENSE`; the MiVOLO copy that ships with its code is `core/perception/mivolo_vendor/LICENSE`.

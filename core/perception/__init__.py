"""
core.perception: pretrained vision models shared by the image and video packages.

    detector.ObjectDetector        RF-DETR Small (COCO classes: people, animals, vehicles, everyday objects)
    recognizer.ZeroShotRecognizer  SigLIP 2 Base (scenes, animal species, vehicle types, genre, time of day)
    face_attributes.FaceAttributes expression (OpenCV zoo MobileFaceNet FER) and identity embeddings (SFace)
    colors.name_colors             dominant colours with human names (CIELAB nearest-name; no model needed)

The large models are downloaded once into the Hugging Face cache by ``python -m services.fetch_models``; the small ONNX
files live in ``core/models``. Every model degrades to "unavailable" rather than raising, and ``status()`` says which.
"""

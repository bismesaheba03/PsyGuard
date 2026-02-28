"""
multimodal.py
CLIP-based image-text mismatch detection.
Detects clickbait thumbnails and image/caption contradictions.
"""

import torch
import requests
from io import BytesIO
from PIL import Image

_model     = None
_processor = None
_available = False


def _load_clip():
    global _model, _processor, _available
    try:
        from transformers import CLIPModel, CLIPProcessor
        print("[CLIP] Loading openai/clip-vit-base-patch32 ...")
        _model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        _processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        _available = True
        print("[CLIP] Ready!")
    except Exception as exc:
        print(f"[CLIP] Not available: {exc}")
        _available = False


_load_clip()


def _fetch_image(url: str) -> Image.Image:
    resp = requests.get(url, timeout=6)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGB")


def analyze_image_caption(image_url: str, caption: str) -> dict:
    """Check whether an image matches its caption (mismatch = potential manipulation)."""
    if not _available or not image_url:
        return {"available": False}

    try:
        image   = _fetch_image(image_url)
        texts   = [caption, f"image completely unrelated to: {caption}"]
        inputs  = _processor(text=texts, images=image, return_tensors="pt", padding=True)

        with torch.no_grad():
            probs = _model(**inputs).logits_per_image.softmax(dim=1)[0]

        match_pct    = round(float(probs[0]) * 100, 1)
        mismatch_pct = round(float(probs[1]) * 100, 1)
        verdict      = ("HIGH MISMATCH" if mismatch_pct > 60
                        else "MODERATE"  if mismatch_pct > 35
                        else "ALIGNED")

        return {
            "available":          True,
            "match_probability":  match_pct,
            "mismatch_score":     mismatch_pct,
            "verdict":            verdict
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}


def analyze_clickbait(image_url: str) -> dict:
    """Detect clickbait visual patterns using CLIP zero-shot."""
    if not _available or not image_url:
        return {"available": False}

    clickbait_labels = [
        "shocking or sensational image designed to provoke outrage",
        "exaggerated facial expression of shock or disbelief",
        "misleading or deceptive visual thumbnail",
        "normal neutral informative image"
    ]

    try:
        image  = _fetch_image(image_url)
        inputs = _processor(text=clickbait_labels, images=image,
                            return_tensors="pt", padding=True)

        with torch.no_grad():
            probs = _model(**inputs).logits_per_image.softmax(dim=1)[0]

        clickbait_score = round(float(probs[:3].sum()) * 100, 1)

        return {
            "available":      True,
            "clickbait_score": clickbait_score,
            "label_scores":   {
                label: round(float(p) * 100, 1)
                for label, p in zip(clickbait_labels, probs)
            }
        }
    except Exception as exc:
        return {"available": False, "error": str(exc)}
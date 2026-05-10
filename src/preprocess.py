from __future__ import annotations

from pathlib import Path
import re

import cv2
import numpy as np

try:
    from .detector import PlateDetection
except ImportError:
    from detector import PlateDetection


PLATE_ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


def load_image(image_path: str | Path) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Gorsel bulunamadi: {path}")

    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Gorsel okunamadi veya desteklenmeyen format: {path}")
    return image


def crop_plate(image: np.ndarray, detection: PlateDetection, padding_ratio: float = 0.06) -> np.ndarray:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = detection.xyxy
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    pad_x = int(box_width * padding_ratio)
    pad_y = int(box_height * padding_ratio)

    left = max(x1 - pad_x, 0)
    top = max(y1 - pad_y, 0)
    right = min(x2 + pad_x, width)
    bottom = min(y2 + pad_y, height)
    return image[top:bottom, left:right].copy()


def build_ocr_variants(plate_image: np.ndarray) -> list[np.ndarray]:
    """Create a small set of OCR-friendly variants without losing the original crop."""
    if plate_image.size == 0:
        return []

    variants = [plate_image]
    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, d=7, sigmaColor=50, sigmaSpace=50)
    threshold = cv2.adaptiveThreshold(
        denoised,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        5,
    )

    variants.extend([gray, enhanced, threshold])
    return variants


def normalize_plate_text(text: str) -> str:
    """Normalize OCR output to a Turkish plate-like uppercase token."""
    normalized = text.upper()
    replacements = {
        "İ": "I",
        "İ": "I",
        "Ş": "S",
        "Ğ": "G",
        "Ü": "U",
        "Ö": "O",
        "Ç": "C",
    }
    for source, target in replacements.items():
        normalized = normalized.replace(source, target)

    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    return normalized


def plate_status(text: str, ocr_confidence: float, min_ocr_confidence: float) -> str:
    if not text:
        return "ocr_okunamadi"
    if ocr_confidence < min_ocr_confidence:
        return "dusuk_guven"
    return "basarili"

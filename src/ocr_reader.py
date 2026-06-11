from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import re
import unicodedata

import numpy as np

try:
    from .preprocess import PLATE_ALLOWED_CHARS, normalize_plate_text
except ImportError:
    from preprocess import PLATE_ALLOWED_CHARS, normalize_plate_text


UC3M_OCR_CLASSES = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass(frozen=True)
class OCRCandidate:
    text: str
    confidence: float
    raw_text: str
    source: str = "none"
    notes: tuple[str, ...] = ()


class EasyOCRPlateReader:
    """EasyOCR wrapper specialized for license plate text."""

    def __init__(self, languages: list[str] | None = None, gpu: bool = False) -> None:
        try:
            import easyocr
        except ImportError as exc:
            raise ImportError("easyocr paketi kurulu degil. `pip install -r requirements.txt` calistirin.") from exc

        self.languages = languages or ["en"]
        self.reader = easyocr.Reader(self.languages, gpu=gpu)

    def read_best(self, image_variants: list[np.ndarray]) -> OCRCandidate:
        candidates = self.read_candidates(image_variants)
        if not candidates:
            return OCRCandidate(text="", confidence=0.0, raw_text="", source="none")

        candidates.sort(key=lambda item: (item.confidence, len(item.text)), reverse=True)
        return candidates[0]

    def read_candidates(self, image_variants: list[np.ndarray]) -> list[OCRCandidate]:
        candidates: list[OCRCandidate] = []
        for variant in image_variants:
            candidates.extend(self._read_variant(variant))
            candidates.extend(self._read_variant(variant, {"mag_ratio": 2.0}))
            candidates.extend(
                self._read_variant(
                    variant,
                    {
                        "text_threshold": 0.4,
                        "low_text": 0.2,
                        "link_threshold": 0.2,
                    },
                )
            )
        return candidates

    def _read_variant(self, image: np.ndarray, options: dict[str, float] | None = None) -> list[OCRCandidate]:
        results = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=PLATE_ALLOWED_CHARS,
            **(options or {}),
        )

        candidates: list[OCRCandidate] = []
        for item in results:
            if len(item) < 3:
                continue
            raw_text = str(item[1])
            confidence = float(item[2])
            cleaned = normalize_plate_text(raw_text)
            if cleaned:
                candidates.append(
                    OCRCandidate(
                        text=cleaned,
                        confidence=confidence,
                        raw_text=raw_text,
                        source="easyocr",
                        notes=("read=easyocr",),
                    )
                )
        return candidates


@dataclass(frozen=True)
class CharacterDetection:
    char: str
    confidence: float
    center_x: float


class YOLOPlateReader:
    """YOLO wrapper that reads a cropped plate by detecting its characters."""

    def __init__(self, weights_path: str | Path, confidence: float = 0.25, image_size: int = 256) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Okuma modeli agirligi bulunamadi: {self.weights_path}. "
                "Kaggle egitiminden sonra best.pt dosyasini models/plate_reader.pt olarak kaydedin."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics paketi kurulu degil. `pip install -r requirements.txt` calistirin.") from exc

        self.model = YOLO(str(self.weights_path))
        self.confidence = confidence
        self.image_size = image_size

    def read(self, plate_image: np.ndarray) -> OCRCandidate:
        if plate_image.size == 0:
            return OCRCandidate(text="", confidence=0.0, raw_text="", source="none")

        results = self.model.predict(
            source=plate_image,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
        )
        if not results:
            return OCRCandidate(text="", confidence=0.0, raw_text="", source="none")

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return OCRCandidate(text="", confidence=0.0, raw_text="", source="none")

        names = self._names()
        characters: list[CharacterDetection] = []
        for box in result.boxes:
            class_id = int(box.cls[0].detach().cpu().item())
            char = self._class_name(class_id, names)
            if not char:
                continue

            xyxy = box.xyxy[0].detach().cpu().numpy()
            center_x = float((xyxy[0] + xyxy[2]) / 2)
            confidence = float(box.conf[0].detach().cpu().item())
            characters.append(CharacterDetection(char=char, confidence=confidence, center_x=center_x))

        if not characters:
            return OCRCandidate(text="", confidence=0.0, raw_text="", source="none")

        characters.sort(key=lambda item: item.center_x)
        raw_text = "".join(item.char for item in characters)
        average_confidence = sum(item.confidence for item in characters) / len(characters)
        text = normalize_plate_text(raw_text)
        if not text:
            return OCRCandidate(
                text="",
                confidence=average_confidence,
                raw_text=raw_text,
                source="yolo",
                notes=(f"read=yolo_chars:{len(characters)}", "normalized_empty"),
            )

        return OCRCandidate(
            text=text,
            confidence=average_confidence,
            raw_text=raw_text,
            source="yolo",
            notes=(f"read=yolo_chars:{len(characters)}",),
        )

    def _class_name(self, class_id: int, names: dict[int, str]) -> str:
        raw_name = names.get(class_id)
        if raw_name is None and 0 <= class_id < len(UC3M_OCR_CLASSES):
            raw_name = UC3M_OCR_CLASSES[class_id]
        if raw_name is None:
            return ""

        normalized = unicodedata.normalize("NFKD", str(raw_name).upper())
        normalized = normalized.encode("ascii", "ignore").decode("ascii")
        cleaned = re.sub(r"[^A-Z0-9]", "", normalized)
        if len(cleaned) != 1 or cleaned not in PLATE_ALLOWED_CHARS:
            return ""
        return cleaned

    def _names(self) -> dict[int, str]:
        raw_names = getattr(self.model, "names", {}) or {}
        if isinstance(raw_names, dict):
            return {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, Iterable):
            return {index: str(value) for index, value in enumerate(raw_names)}
        return {}

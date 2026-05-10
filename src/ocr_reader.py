from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from .preprocess import PLATE_ALLOWED_CHARS, normalize_plate_text
except ImportError:
    from preprocess import PLATE_ALLOWED_CHARS, normalize_plate_text


@dataclass(frozen=True)
class OCRCandidate:
    text: str
    confidence: float
    raw_text: str


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
        candidates: list[OCRCandidate] = []
        for variant in image_variants:
            candidates.extend(self._read_variant(variant))

        if not candidates:
            return OCRCandidate(text="", confidence=0.0, raw_text="")

        candidates.sort(key=lambda item: (item.confidence, len(item.text)), reverse=True)
        return candidates[0]

    def _read_variant(self, image: np.ndarray) -> list[OCRCandidate]:
        results = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist=PLATE_ALLOWED_CHARS,
        )

        candidates: list[OCRCandidate] = []
        for item in results:
            if len(item) < 3:
                continue
            raw_text = str(item[1])
            confidence = float(item[2])
            cleaned = normalize_plate_text(raw_text)
            if cleaned:
                candidates.append(OCRCandidate(text=cleaned, confidence=confidence, raw_text=raw_text))
        return candidates

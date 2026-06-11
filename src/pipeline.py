from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
from threading import Lock

import numpy as np

from .detector import PlateDetection, PlateDetector
from .ocr_reader import EasyOCRPlateReader, OCRCandidate, YOLOPlateReader
from .preprocess import (
    build_ocr_variants,
    crop_plate,
    load_image,
    noisy_plate_candidates,
    normalize_plate_text,
    plate_status,
)
from .visualize import draw_plate_result, save_image


@dataclass(frozen=True)
class SinglePlateResult:
    text: str
    raw_text: str
    status: str
    reader_source: str
    reader_confidence: float
    detection_confidence: float
    bbox: tuple[int, int, int, int]
    crop_path: Path


@dataclass(frozen=True)
class SingleImageResult:
    image_path: Path
    annotated_path: Path
    plate: SinglePlateResult | None
    status: str
    message: str

    @property
    def text(self) -> str:
        if self.plate is None:
            return ""
        return self.plate.text


class SingleImagePlatePipeline:
    """Process one vehicle image and return UI-friendly output paths."""

    def __init__(
        self,
        detector_weights: str | Path = "models/plate_detector.pt",
        reader_weights: str | Path = "models/plate_reader.pt",
        output_dir: str | Path = "outputs/web",
        confidence: float = 0.25,
        reader_confidence: float = 0.25,
        ocr_confidence: float = 0.50,
        image_size: int = 640,
        reader_image_size: int = 256,
        reader_mode: str = "hybrid",
        easyocr_gpu: bool = False,
        disable_easyocr: bool = False,
    ) -> None:
        if reader_mode not in {"hybrid", "easyocr", "yolo"}:
            raise ValueError("reader_mode değeri hybrid, easyocr veya yolo olmalı.")

        self.detector_weights = Path(detector_weights)
        self.reader_weights = Path(reader_weights)
        self.output_dir = Path(output_dir)
        self.confidence = confidence
        self.reader_confidence = reader_confidence
        self.ocr_confidence = ocr_confidence
        self.image_size = image_size
        self.reader_image_size = reader_image_size
        self.reader_mode = reader_mode
        self.easyocr_gpu = easyocr_gpu
        self.disable_easyocr = disable_easyocr

        self._detector: PlateDetector | None = None
        self._yolo_reader: YOLOPlateReader | None = None
        self._easyocr_reader: EasyOCRPlateReader | None = None
        self._lock = Lock()

    def process(self, image_path: str | Path) -> SingleImageResult:
        with self._lock:
            return self._process_unlocked(Path(image_path))

    def _process_unlocked(self, image_path: Path) -> SingleImageResult:
        image = load_image(image_path)
        annotated_path = self.output_dir / "annotated" / f"{image_path.stem}_result.jpg"
        detections = self._get_detector().detect(image, max_plates=1)

        if not detections:
            annotated = draw_plate_result(image, None, "", "tespit_edilemedi")
            save_image(annotated_path, annotated)
            return SingleImageResult(
                image_path=image_path,
                annotated_path=annotated_path,
                plate=None,
                status="tespit_edilemedi",
                message="Plaka tespit edilemedi.",
            )

        detection = detections[0]
        plate_crop = crop_plate(image, detection)
        crop_path = self.output_dir / "crops" / f"{image_path.stem}_plate.jpg"
        save_image(crop_path, plate_crop)

        candidate = self._read_plate(plate_crop)
        status = plate_status(candidate.text, candidate.confidence, self.ocr_confidence)
        annotated = draw_plate_result(image, detection, candidate.text, status)
        save_image(annotated_path, annotated)

        plate = SinglePlateResult(
            text=candidate.text,
            raw_text=candidate.raw_text,
            status=status,
            reader_source=candidate.source,
            reader_confidence=candidate.confidence,
            detection_confidence=detection.confidence,
            bbox=detection.xyxy,
            crop_path=crop_path,
        )
        return SingleImageResult(
            image_path=image_path,
            annotated_path=annotated_path,
            plate=plate,
            status=status,
            message=_status_message(status, candidate.text),
        )

    def _get_detector(self) -> PlateDetector:
        if self._detector is None:
            self._detector = PlateDetector(
                self.detector_weights,
                confidence=self.confidence,
                image_size=self.image_size,
            )
        return self._detector

    def _get_yolo_reader(self) -> YOLOPlateReader:
        if self._yolo_reader is None:
            self._yolo_reader = YOLOPlateReader(
                self.reader_weights,
                confidence=self.reader_confidence,
                image_size=self.reader_image_size,
            )
        return self._yolo_reader

    def _get_easyocr_reader(self) -> EasyOCRPlateReader:
        if self._easyocr_reader is None:
            self._easyocr_reader = EasyOCRPlateReader(gpu=self.easyocr_gpu)
        return self._easyocr_reader

    def _read_plate(self, plate_crop: np.ndarray) -> OCRCandidate:
        if self.reader_mode == "easyocr":
            easyocr_candidates = self._get_easyocr_reader().read_candidates(build_ocr_variants(plate_crop))
            return choose_easyocr_only_candidate(easyocr_candidates)

        yolo_candidate = self._get_yolo_reader().read(plate_crop)
        if self.reader_mode == "yolo" or self.disable_easyocr:
            return yolo_candidate

        if should_try_easyocr(yolo_candidate, self.ocr_confidence):
            easyocr_candidates = self._get_easyocr_reader().read_candidates(build_ocr_variants(plate_crop))
            return choose_ocr_candidate(yolo_candidate, easyocr_candidates, self.ocr_confidence)

        return yolo_candidate


def _status_message(status: str, text: str) -> str:
    if status == "basarili":
        return f"Plaka okundu: {text}"
    if status == "review_needed":
        return f"Plaka okundu: {text}"
    return "Plaka tespit edildi ancak OCR okunamadı."


def should_try_easyocr(candidate: OCRCandidate, min_confidence: float) -> bool:
    if not candidate.text or candidate.confidence < min_confidence:
        return True
    return bool(candidate.raw_text and candidate.raw_text != candidate.text)


def choose_easyocr_only_candidate(easyocr_candidates: list[OCRCandidate]) -> OCRCandidate:
    pool = [candidate for candidate in easyocr_candidates if candidate.text]
    if not pool:
        return OCRCandidate(text="", confidence=0.0, raw_text="", source="none")

    counts = Counter(candidate.text for candidate in pool)
    raw_texts = [candidate.raw_text for candidate in pool if candidate.raw_text]
    return max(
        pool,
        key=lambda candidate: score_candidate(
            candidate,
            0.0,
            counts,
            set(),
            set(),
            "",
            raw_texts,
        ),
    )


def choose_ocr_candidate(
    yolo_candidate: OCRCandidate,
    easyocr_candidates: list[OCRCandidate],
    min_confidence: float,
) -> OCRCandidate:
    if yolo_candidate.text and yolo_candidate.confidence >= min_confidence:
        has_clean_raw = not yolo_candidate.raw_text or yolo_candidate.raw_text == yolo_candidate.text
        if has_clean_raw:
            return yolo_candidate

    province_hints = province_hints_from_raw(yolo_candidate.raw_text)
    pool: list[tuple[OCRCandidate, float]] = []
    if yolo_candidate.text:
        pool.append((yolo_candidate, 0.0))
    pool.extend((candidate, 0.0) for candidate in easyocr_candidates if candidate.text)
    pool.extend(build_province_fusion_candidates(easyocr_candidates, province_hints))
    yolo_repairs = build_yolo_repair_candidates(yolo_candidate)
    pool.extend(yolo_repairs)

    if not pool:
        return yolo_candidate

    easyocr_counts = Counter(candidate.text for candidate in easyocr_candidates if candidate.text)
    yolo_repair_texts = {candidate.text for candidate, _ in yolo_repairs}
    easyocr_raw_texts = [candidate.raw_text for candidate in easyocr_candidates if candidate.raw_text]

    best_candidate = max(
        pool,
        key=lambda item: score_candidate(
            item[0],
            item[1],
            easyocr_counts,
            yolo_repair_texts,
            province_hints,
            yolo_candidate.raw_text,
            easyocr_raw_texts,
        ),
    )[0]

    if best_candidate.text:
        return best_candidate
    return yolo_candidate


def build_yolo_repair_candidates(yolo_candidate: OCRCandidate) -> list[tuple[OCRCandidate, float]]:
    if not yolo_candidate.raw_text:
        return []

    repairs: list[tuple[OCRCandidate, float]] = []
    for cost, text in noisy_plate_candidates(yolo_candidate.raw_text):
        if cost > 3.2:
            continue
        confidence = max(0.01, yolo_candidate.confidence - cost * 0.05)
        repairs.append(
            (
                OCRCandidate(
                    text=text,
                    confidence=confidence,
                    raw_text=yolo_candidate.raw_text,
                    source="yolo",
                ),
                cost,
            )
        )
    return repairs


def build_province_fusion_candidates(
    easyocr_candidates: list[OCRCandidate],
    province_hints: set[str],
) -> list[tuple[OCRCandidate, float]]:
    if not province_hints:
        return []

    fused: dict[str, OCRCandidate] = {}
    for easyocr_candidate in easyocr_candidates:
        if not easyocr_candidate.text or len(easyocr_candidate.text) < 5:
            continue
        suffix = easyocr_candidate.text[2:]
        for province in province_hints:
            text = normalize_plate_text(province + suffix)
            if not text:
                continue
            confidence = max(0.01, easyocr_candidate.confidence - 0.05)
            candidate = OCRCandidate(
                text=text,
                confidence=confidence,
                raw_text=easyocr_candidate.raw_text,
                source="easyocr",
            )
            previous = fused.get(text)
            if previous is None or candidate.confidence > previous.confidence:
                fused[text] = candidate
    return [(candidate, 0.0) for candidate in fused.values()]


def score_candidate(
    candidate: OCRCandidate,
    repair_cost: float,
    easyocr_counts: Counter[str],
    yolo_repair_texts: set[str],
    province_hints: set[str],
    yolo_raw_text: str,
    easyocr_raw_texts: list[str],
) -> float:
    score = candidate.confidence
    if candidate.text[:2] in province_hints:
        score += 0.45
    score += min(easyocr_counts.get(candidate.text, 0), 4) * 0.08
    if candidate.text in yolo_repair_texts and easyocr_counts.get(candidate.text, 0):
        score += 0.75
    score += raw_support(candidate.text, yolo_raw_text) * 0.25
    score += max((raw_support(candidate.text, raw_text) for raw_text in easyocr_raw_texts), default=0.0) * 0.90
    score += plate_shape_score(candidate.text)
    if candidate.source == "easyocr" and candidate.confidence < 0.50:
        score -= 0.35
    if candidate.source == "yolo" and repair_cost:
        score += 0.18
        score -= repair_cost * 0.03
    return score


def plate_shape_score(text: str) -> float:
    if len(text) < 5:
        return 0.0

    best: float | None = None
    for letter_count in range(1, 4):
        digits = text[2 + letter_count :]
        letters = text[2 : 2 + letter_count]
        if not letters.isalpha() or not digits.isdigit():
            continue
        if not 2 <= len(digits) <= 4:
            continue
        score = 0.0
        if letter_count == 3 and len(digits) == 3:
            score += 0.10
        if letter_count == 2 and len(digits) == 3:
            score -= 0.05
        if any(left == right for left, right in zip(letters, letters[1:])):
            score -= 0.12
        if len(set(letters)) < len(letters):
            score -= 0.08
        best = score if best is None else max(best, score)
    return best or 0.0


def province_hints_from_raw(raw_text: str) -> set[str]:
    cleaned = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    hints: set[str] = set()
    for first in range(min(len(cleaned), 4)):
        for second in range(first + 1, min(len(cleaned), 5)):
            chars = cleaned[first] + cleaned[second]
            digits = "".join(coerce_digit(char) or "" for char in chars)
            if len(digits) != 2:
                continue
            province_number = int(digits)
            if 1 <= province_number <= 81:
                hints.add(digits)
    return hints


def coerce_digit(char: str) -> str | None:
    if char.isdigit():
        return char
    return {
        "B": "8",
        "D": "0",
        "G": "6",
        "I": "1",
        "L": "1",
        "O": "0",
        "Q": "0",
        "S": "5",
        "T": "7",
        "Z": "2",
    }.get(char)


def raw_support(candidate_text: str, raw_text: str) -> float:
    cleaned_raw = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if not candidate_text or not cleaned_raw:
        return 0.0
    return longest_common_subsequence(candidate_text, cleaned_raw) / len(candidate_text)


def longest_common_subsequence(left: str, right: str) -> int:
    previous = [0] * (len(right) + 1)
    for left_char in left:
        current = [0]
        for index, right_char in enumerate(right, start=1):
            if left_char == right_char:
                current.append(previous[index - 1] + 1)
            else:
                current.append(max(previous[index], current[-1]))
        previous = current
    return previous[-1]

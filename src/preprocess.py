from __future__ import annotations

from itertools import combinations, product
from pathlib import Path
import re
import unicodedata

import cv2
import numpy as np

try:
    from .detector import PlateDetection
except ImportError:
    from detector import PlateDetection


PLATE_ALLOWED_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
PLATE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
PLATE_DIGITS = "0123456789"

LETTER_FIXES = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
}
DIGIT_FIXES = {
    "A": "4",
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
}
LETTER_CONFUSIONS = {
    "B": "E",
    "F": "E",
    "H": "M",
    "N": "M",
    "W": "M",
}


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
    blue_band_removed = remove_left_blue_band(plate_image)
    if blue_band_removed is not None:
        variants.append(blue_band_removed)

    processed: list[np.ndarray] = []
    for variant in variants:
        processed.extend(_build_single_ocr_variants(variant))

    return variants + processed


def remove_left_blue_band(plate_image: np.ndarray) -> np.ndarray | None:
    """Crop the Turkish plate blue strip when it is clearly visible on the left."""
    if plate_image.ndim != 3 or plate_image.shape[1] < 40:
        return None

    height, width = plate_image.shape[:2]
    search_width = max(int(width * 0.35), 1)
    hsv = cv2.cvtColor(plate_image[:, :search_width], cv2.COLOR_BGR2HSV)
    blue_mask = cv2.inRange(hsv, (90, 45, 30), (135, 255, 255))
    blue_ratio_by_column = (blue_mask > 0).mean(axis=0)
    blue_columns = np.where(blue_ratio_by_column > 0.12)[0]
    if len(blue_columns) == 0:
        return None

    cut_x = int(blue_columns[-1]) + 2
    min_remaining_width = max(24, int(width * 0.60))
    if width - cut_x < min_remaining_width:
        return None

    return plate_image[:, cut_x:].copy()


def _build_single_ocr_variants(plate_image: np.ndarray) -> list[np.ndarray]:
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

    return [gray, enhanced, threshold]


def normalize_plate_text(text: str) -> str:
    """Normalize OCR output and keep only valid Turkish plate forms."""
    normalized = unicodedata.normalize("NFKD", text.upper())
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^A-Z0-9]", "", normalized)
    return normalize_turkish_plate(normalized)


def normalize_turkish_plate(text: str) -> str:
    """Return a compact valid Turkish plate text, or an empty string."""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not cleaned:
        return ""

    candidates: list[tuple[int, str]] = []
    for letter_count in range(1, 4):
        for digit_count in range(2, 5):
            if len(cleaned) != 2 + letter_count + digit_count:
                continue

            province = _coerce_section(cleaned[:2], PLATE_DIGITS, DIGIT_FIXES)
            letters = _coerce_section(cleaned[2 : 2 + letter_count], PLATE_LETTERS, LETTER_FIXES)
            digits = _coerce_section(cleaned[2 + letter_count :], PLATE_DIGITS, DIGIT_FIXES)
            if province is None or letters is None or digits is None:
                continue

            province_text, province_cost = province
            province_number = int(province_text)
            if not 1 <= province_number <= 81:
                continue

            letters_text, letters_cost = letters
            digits_text, digits_cost = digits
            candidate = f"{province_text}{letters_text}{digits_text}"
            candidates.append((province_cost + letters_cost + digits_cost, candidate))

    if not candidates:
        return ""

    candidates.sort(key=lambda item: (item[0], item[1]))
    return candidates[0][1]


def noisy_plate_candidates(text: str, max_deletions: int = 3) -> list[tuple[float, str]]:
    """Return Turkish plate candidates from noisy OCR text with small edits."""
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if not cleaned:
        return []

    candidates: dict[str, float] = {}
    min_length = 5
    max_length = min(9, len(cleaned))
    for target_length in range(min_length, max_length + 1):
        deletion_count = len(cleaned) - target_length
        if deletion_count > max_deletions:
            continue

        for indexes in combinations(range(len(cleaned)), target_length):
            selected = "".join(cleaned[index] for index in indexes)
            deletion_cost = deletion_count * 0.8
            for letter_count in range(1, 4):
                for digit_count in range(2, 5):
                    if target_length != 2 + letter_count + digit_count:
                        continue
                    _add_noisy_candidate(candidates, selected, letter_count, digit_count, deletion_cost)

    return sorted((cost, candidate) for candidate, cost in candidates.items())


def is_valid_turkish_plate(text: str) -> bool:
    return normalize_turkish_plate(text) == re.sub(r"[^A-Z0-9]", "", text.upper())


def _add_noisy_candidate(
    candidates: dict[str, float],
    selected: str,
    letter_count: int,
    digit_count: int,
    deletion_cost: float,
) -> None:
    slots = ["digit", "digit"] + ["letter"] * letter_count + ["digit"] * digit_count
    option_groups = [_slot_options(char, slot) for char, slot in zip(selected, slots)]
    if any(not group for group in option_groups):
        return

    for options in product(*option_groups):
        chars = [option[0] for option in options]
        costs = [option[1] for option in options]
        first_letter_from_digit = options[2][2]
        province = "".join(chars[:2])
        province_number = int(province)
        if not 1 <= province_number <= 81:
            continue

        candidate = "".join(chars)
        cost = deletion_cost + sum(costs)
        if digit_count == 4:
            cost += 0.45
        if first_letter_from_digit:
            cost += 0.85

        previous = candidates.get(candidate)
        if previous is None or cost < previous:
            candidates[candidate] = cost


def _slot_options(char: str, slot: str) -> list[tuple[str, float, bool]]:
    if slot == "digit":
        if char in PLATE_DIGITS:
            return [(char, 0.0, False)]
        replacement = DIGIT_FIXES.get(char)
        if replacement is None:
            return []
        return [(replacement, 1.05, False)]

    options: list[tuple[str, float, bool]] = []
    if char in PLATE_LETTERS:
        options.append((char, 0.0, False))
        replacement = LETTER_CONFUSIONS.get(char)
        if replacement is not None:
            options.append((replacement, 0.65, False))
    replacement = LETTER_FIXES.get(char)
    if replacement is not None:
        options.append((replacement, 0.75, char in PLATE_DIGITS))
    return options


def _coerce_section(
    section: str,
    allowed_chars: str,
    replacements: dict[str, str],
) -> tuple[str, int] | None:
    output: list[str] = []
    cost = 0
    for char in section:
        if char in allowed_chars:
            output.append(char)
            continue
        replacement = replacements.get(char)
        if replacement is None:
            return None
        output.append(replacement)
        cost += 1
    return "".join(output), cost


def plate_status(text: str, ocr_confidence: float, min_ocr_confidence: float) -> str:
    if not text:
        return "ocr_okunamadi"
    if ocr_confidence < min_ocr_confidence:
        return "review_needed"
    return "basarili"

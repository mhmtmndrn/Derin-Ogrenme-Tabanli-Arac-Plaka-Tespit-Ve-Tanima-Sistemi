from __future__ import annotations

"""Batch runner for plate detection, OCR, visualization, and HTML/CSV reporting."""

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from html import escape
from pathlib import Path

import cv2
import numpy as np

from src.detector import PlateDetection, PlateDetector
from src.ocr_reader import EasyOCRPlateReader, OCRCandidate, YOLOPlateReader
from src.preprocess import (
    build_ocr_variants,
    crop_plate,
    load_image,
    noisy_plate_candidates,
    normalize_plate_text,
    plate_status,
)
from src.visualize import save_image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
# The CSV report mirrors the per-plate rows that the HTML gallery renders.
CSV_FIELDS = [
    "image_name",
    "image_path",
    "plate_index",
    "detected_text",
    "raw_text",
    "reader_source",
    "reader_confidence",
    "detection_confidence",
    "status",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "crop_path",
    "annotated_path",
]


@dataclass(frozen=True)
class PlateRead:
    # A single detection plus the text chosen for that crop.
    detection: PlateDetection
    text: str
    raw_text: str
    reader_source: str
    reader_confidence: float
    status: str
    crop_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tum araba gorsellerinde iki asamali plaka tespit ve okuma.")
    parser.add_argument("--images", default="arabalar", help="Gorsellerin bulundugu klasor.")
    parser.add_argument("--detector-weights", default="models/plate_detector.pt", help="Plaka tespit modeli.")
    parser.add_argument("--reader-weights", default="models/plate_reader.pt", help="Plaka okuma/karakter modeli.")
    parser.add_argument("--output-dir", default="outputs", help="Sonuclarin yazilacagi klasor.")
    parser.add_argument("--conf", type=float, default=0.25, help="Plaka tespit guven esigi.")
    parser.add_argument(
        "--reader-mode",
        choices=["hybrid", "easyocr", "yolo"],
        default="hybrid",
        help="Okuma modu: hybrid=YOLO+EasyOCR, easyocr=sadece EasyOCR, yolo=sadece karakter modeli.",
    )
    parser.add_argument("--reader-conf", type=float, default=0.25, help="Karakter okuma guven esigi.")
    parser.add_argument("--ocr-conf", type=float, default=0.50, help="Basarili okuma icin minimum ortalama guven.")
    parser.add_argument("--easyocr-gpu", action="store_true", help="EasyOCR fallback icin GPU kullan.")
    parser.add_argument("--disable-easyocr", action="store_true", help="EasyOCR fallback okumasini kapat.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tespit modeli girdi boyutu.")
    parser.add_argument("--reader-imgsz", type=int, default=256, help="Okuma modeli girdi boyutu.")
    parser.add_argument("--max-plates", type=int, default=3, help="Her gorselde okunacak maksimum plaka sayisi.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_dir = Path(args.images)
    output_dir = Path(args.output_dir)
    annotated_dir = output_dir / "annotated"
    crop_dir = output_dir / "crops"
    csv_path = output_dir / "all_results.csv"

    # Sort input files naturally so image_2.jpg comes before image_10.jpg.
    image_paths = sorted(
        [path for path in image_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS],
        key=natural_sort_key,
    )
    if not image_paths:
        raise FileNotFoundError(f"Gorsel bulunamadi: {image_dir}")

    # Load the detection model once and reuse it for the full batch.
    detector = PlateDetector(args.detector_weights, confidence=args.conf, image_size=args.imgsz)
    reader: YOLOPlateReader | None = None
    easyocr_reader: EasyOCRPlateReader | None = None
    if args.reader_mode != "easyocr":
        reader = YOLOPlateReader(args.reader_weights, confidence=args.reader_conf, image_size=args.reader_imgsz)
    if args.reader_mode == "easyocr":
        easyocr_reader = EasyOCRPlateReader(gpu=args.easyocr_gpu)

    rows: list[dict[str, str]] = []
    for image_index, image_path in enumerate(image_paths, start=1):
        print(f"[{image_index}/{len(image_paths)}] {image_path.name}")
        image = load_image(image_path)
        detections = detector.detect(image, max_plates=args.max_plates)
        annotated_path = annotated_dir / f"{image_path.stem}_result.jpg"

        if not detections:
            # Keep a visible output even when no plate is found.
            annotated = draw_no_detection(image)
            save_image(annotated_path, annotated)
            rows.append(no_detection_row(image_path, annotated_path))
            continue

        reads: list[PlateRead] = []
        for plate_index, detection in enumerate(detections, start=1):
            # Each plate crop is stored separately so OCR failures can be inspected later.
            plate_crop = crop_plate(image, detection)
            crop_path = crop_dir / f"{image_path.stem}_plate_{plate_index}.jpg"
            save_image(crop_path, plate_crop)

            if args.reader_mode == "easyocr":
                if easyocr_reader is None:
                    easyocr_reader = EasyOCRPlateReader(gpu=args.easyocr_gpu)
                easyocr_candidates = easyocr_reader.read_candidates(build_ocr_variants(plate_crop))
                candidate = choose_easyocr_only_candidate(easyocr_candidates)
            else:
                if reader is None:
                    raise RuntimeError("YOLO okuma modeli yuklenmedi.")
                candidate = reader.read(plate_crop)
                if args.reader_mode == "hybrid" and should_try_easyocr(candidate, args.ocr_conf) and not args.disable_easyocr:
                    if easyocr_reader is None:
                        easyocr_reader = EasyOCRPlateReader(gpu=args.easyocr_gpu)
                    easyocr_candidates = easyocr_reader.read_candidates(build_ocr_variants(plate_crop))
                    candidate = choose_ocr_candidate(candidate, easyocr_candidates, args.ocr_conf)

            # Normalize the final status after the best candidate is selected.
            status = plate_status(candidate.text, candidate.confidence, args.ocr_conf)
            reads.append(
                PlateRead(
                    detection=detection,
                    text=candidate.text,
                    raw_text=candidate.raw_text,
                    reader_source=candidate.source,
                    reader_confidence=candidate.confidence,
                    status=status,
                    crop_path=crop_path,
                )
            )

        annotated = draw_reads(image, reads)
        save_image(annotated_path, annotated)
        for plate_index, read in enumerate(reads, start=1):
            rows.append(read_row(image_path, plate_index, read, annotated_path))

    write_csv(csv_path, rows)
    gallery_path = output_dir / "index.html"
    contact_sheet_path = output_dir / "contact_sheet.jpg"
    write_gallery(gallery_path, rows, output_dir)
    write_contact_sheet(image_paths, annotated_dir, contact_sheet_path)

    print(f"\nToplam gorsel: {len(image_paths)}")
    print(f"CSV: {csv_path}")
    print(f"Galeri: {gallery_path}")
    print(f"Ozet gorsel: {contact_sheet_path}")
    print(f"Isaretli gorseller: {annotated_dir}")
    print(f"Kirpilmis plakalar: {crop_dir}")
    return 0


def natural_sort_key(path: Path) -> list[tuple[int, int | str]]:
    # Break the stem into digit and non-digit chunks for human-friendly ordering.
    parts: list[tuple[int, int | str]] = []
    current = ""
    for char in path.stem:
        if char.isdigit():
            current += char
        else:
            if current:
                parts.append((0, int(current)))
                current = ""
            parts.append((1, char.lower()))
    if current:
        parts.append((0, int(current)))
    parts.append((1, path.suffix.lower()))
    return parts


def should_try_easyocr(candidate: OCRCandidate, min_confidence: float) -> bool:
    # Hybrid mode falls back when YOLO text is weak or looks suspicious.
    if not candidate.text or candidate.confidence < min_confidence:
        return True
    return bool(candidate.raw_text and candidate.raw_text != candidate.text)


def choose_ocr_candidate(
    yolo_candidate: OCRCandidate,
    easyocr_candidates: list[OCRCandidate],
    min_confidence: float,
) -> OCRCandidate:
    # Blend the YOLO result with EasyOCR and heuristic repairs, then score the pool.
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


def choose_easyocr_only_candidate(easyocr_candidates: list[OCRCandidate]) -> OCRCandidate:
    # EasyOCR-only mode still ranks multiple variants before choosing one text.
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


def build_yolo_repair_candidates(yolo_candidate: OCRCandidate) -> list[tuple[OCRCandidate, float]]:
    # Generate small OCR edits for the YOLO text when the raw sequence is noisy.
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
    # Recombine EasyOCR suffixes with likely province codes extracted from raw text.
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
    # Score the candidate by combining direct confidence with shape and cross-reader support.
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
    # Turkish plates follow a fairly constrained shape, so reward plausible layouts.
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
    # Extract two-digit province candidates from loose OCR text.
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
    # Map common OCR confusions back to digits when possible.
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
    # Measure how much of the cleaned candidate appears in the raw OCR text.
    cleaned_raw = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if not candidate_text or not cleaned_raw:
        return 0.0
    return longest_common_subsequence(candidate_text, cleaned_raw) / len(candidate_text)


def longest_common_subsequence(left: str, right: str) -> int:
    # Small dynamic-programming helper used to estimate text similarity.
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


def draw_reads(image: np.ndarray, reads: list[PlateRead]) -> np.ndarray:
    # Draw every detection with its selected OCR result.
    canvas = image.copy()
    for index, read in enumerate(reads, start=1):
        x1, y1, x2, y2 = read.detection.xyxy
        color = (0, 180, 0) if read.status == "basarili" else (0, 180, 255)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        text = read.text or "OCR okunamadi"
        label = f"{index}: {text} | {read.reader_source} {read.reader_confidence:.2f}"
        label_y = max(y1 - 10, 22 + (index - 1) * 24)
        draw_label(canvas, label, x1, label_y, color)
    return canvas


def draw_no_detection(image: np.ndarray) -> np.ndarray:
    # Render a simple banner when the detector finds nothing.
    canvas = image.copy()
    draw_label(canvas, "Plaka tespit edilemedi", 12, 30, (0, 0, 255))
    return canvas


def draw_label(image: np.ndarray, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    # Draw a white label box so the text stays readable on any background.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
    thickness = 2
    (width, height), baseline = cv2.getTextSize(label, font, scale, thickness)
    left = max(x, 0)
    top = max(y - height - baseline - 4, 0)
    right = min(left + width + 8, image.shape[1] - 1)
    bottom = min(top + height + baseline + 8, image.shape[0] - 1)
    cv2.rectangle(image, (left, top), (right, bottom), (255, 255, 255), -1)
    cv2.putText(image, label, (left + 4, bottom - baseline - 4), font, scale, color, thickness, cv2.LINE_AA)


def no_detection_row(image_path: Path, annotated_path: Path) -> dict[str, str]:
    # Standardize the CSV row for images where no plate was found.
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "plate_index": "0",
        "detected_text": "",
        "raw_text": "",
        "reader_source": "none",
        "reader_confidence": "0.0000",
        "detection_confidence": "0.0000",
        "status": "tespit_edilemedi",
        "bbox_x1": "",
        "bbox_y1": "",
        "bbox_x2": "",
        "bbox_y2": "",
        "crop_path": "",
        "annotated_path": str(annotated_path),
    }


def read_row(image_path: Path, plate_index: int, read: PlateRead, annotated_path: Path) -> dict[str, str]:
    # Standardize the CSV row for one detected plate.
    x1, y1, x2, y2 = read.detection.xyxy
    return {
        "image_name": image_path.name,
        "image_path": str(image_path),
        "plate_index": str(plate_index),
        "detected_text": read.text,
        "raw_text": read.raw_text,
        "reader_source": read.reader_source,
        "reader_confidence": f"{read.reader_confidence:.4f}",
        "detection_confidence": f"{read.detection.confidence:.4f}",
        "status": read.status,
        "bbox_x1": str(x1),
        "bbox_y1": str(y1),
        "bbox_x2": str(x2),
        "bbox_y2": str(y2),
        "crop_path": str(read.crop_path),
        "annotated_path": str(annotated_path),
    }


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    # Emit UTF-8-sig so spreadsheet tools open the CSV cleanly.
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_gallery(path: Path, rows: list[dict[str, str]], output_dir: Path) -> None:
    # Build a tiny static gallery so results can be inspected without Python.
    path.parent.mkdir(parents=True, exist_ok=True)
    cards = []
    for row in rows:
        annotated_src = html_relative_path(row["annotated_path"], output_dir)
        crop_src = html_relative_path(row["crop_path"], output_dir) if row["crop_path"] else ""
        crop_html = f'<img src="{escape(crop_src)}" alt="crop">' if crop_src else "<div class=\"empty\">kirpim yok</div>"
        cards.append(
            "\n".join(
                [
                    "<article>",
                    f"<h2>{escape(row['image_name'])} / plaka {escape(row['plate_index'])}</h2>",
                    f'<img src="{escape(annotated_src)}" alt="annotated">',
                    crop_html,
                    "<dl>",
                    f"<dt>Okunan</dt><dd>{escape(row['detected_text'] or '-')}</dd>",
                    f"<dt>Ham okuma</dt><dd>{escape(row['raw_text'] or '-')}</dd>",
                    f"<dt>Okuyucu</dt><dd>{escape(row['reader_source'])}</dd>",
                    f"<dt>Durum</dt><dd>{escape(row['status'])}</dd>",
                    f"<dt>Tespit guveni</dt><dd>{escape(row['detection_confidence'])}</dd>",
                    f"<dt>Okuma guveni</dt><dd>{escape(row['reader_confidence'])}</dd>",
                    "</dl>",
                    "</article>",
                ]
            )
        )

    html = f"""<!doctype html>
<html lang="tr">
<head>
  <meta charset="utf-8">
  <title>Plaka Tespit ve Okuma Sonuclari</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f5f5f5; color: #222; }}
    h1 {{ margin: 0 0 18px; font-size: 24px; }}
    main {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(310px, 1fr)); gap: 16px; }}
    article {{ background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    h2 {{ font-size: 16px; margin: 0 0 10px; }}
    img {{ display: block; max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }}
    .empty {{ padding: 20px; border: 1px dashed #bbb; color: #666; text-align: center; }}
    dl {{ display: grid; grid-template-columns: 120px 1fr; gap: 4px 8px; margin: 10px 0 0; font-size: 14px; }}
    dt {{ font-weight: bold; }}
    dd {{ margin: 0; }}
  </style>
</head>
<body>
  <h1>Plaka Tespit ve Okuma Sonuclari</h1>
  <main>
    {"".join(cards)}
  </main>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def html_relative_path(value: str, output_dir: Path) -> str:
    # Keep generated links relative when possible, which makes the gallery portable.
    path = Path(value)
    try:
        path = path.relative_to(output_dir)
    except ValueError:
        pass
    return path.as_posix()


def write_contact_sheet(image_paths: list[Path], annotated_dir: Path, output_path: Path) -> None:
    # Compose a single overview image that fits many annotated results on one page.
    columns = 5
    cell_width = 340
    cell_height = 260
    label_height = 32
    rows = (len(image_paths) + columns - 1) // columns
    sheet = np.full((rows * cell_height, columns * cell_width, 3), 255, dtype=np.uint8)

    for index, image_path in enumerate(image_paths):
        annotated_path = annotated_dir / f"{image_path.stem}_result.jpg"
        image = cv2.imread(str(annotated_path))
        if image is None:
            continue

        row = index // columns
        column = index % columns
        x = column * cell_width
        y = row * cell_height
        max_width = cell_width - 16
        max_height = cell_height - label_height - 16
        scale = min(max_width / image.shape[1], max_height / image.shape[0])
        resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))

        top = y + 8
        left = x + (cell_width - resized.shape[1]) // 2
        sheet[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
        cv2.rectangle(sheet, (x, y), (x + cell_width - 1, y + cell_height - 1), (210, 210, 210), 1)
        cv2.putText(
            sheet,
            image_path.name,
            (x + 10, y + cell_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    save_image(output_path, sheet)


if __name__ == "__main__":
    raise SystemExit(main())

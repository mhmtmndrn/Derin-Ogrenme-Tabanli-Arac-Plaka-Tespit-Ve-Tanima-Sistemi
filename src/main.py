from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parent))

try:
    from .detector import PlateDetector
    from .ocr_reader import EasyOCRPlateReader
    from .preprocess import build_ocr_variants, crop_plate, load_image, plate_status
    from .visualize import draw_plate_result, save_image
except ImportError:
    from detector import PlateDetector
    from ocr_reader import EasyOCRPlateReader
    from preprocess import build_ocr_variants, crop_plate, load_image, plate_status
    from visualize import draw_plate_result, save_image


CSV_FIELDS = [
    "image_path",
    "plate_index",
    "detected_text",
    "ocr_confidence",
    "detection_confidence",
    "status",
    "crop_path",
    "output_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Arac fotografi uzerinden plaka tespit ve OCR prototipi.")
    parser.add_argument("--image", required=True, help="Islenecek arac fotografi.")
    parser.add_argument("--weights", required=True, help="YOLO model agirligi. Ornek: models/best.pt")
    parser.add_argument("--output", default="outputs/demo_result.jpg", help="Isaretlenmis cikti gorseli.")
    parser.add_argument("--csv", default="outputs/results.csv", help="Sonuc tablosu CSV dosyasi.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO tespit guven esigi.")
    parser.add_argument("--ocr-conf", type=float, default=0.50, help="Basarili OCR icin minimum guven esigi.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO tahmin girdi boyutu.")
    parser.add_argument("--max-plates", type=int, default=1, help="Islenecek maksimum plaka sayisi.")
    parser.add_argument("--gpu", action="store_true", help="EasyOCR icin GPU kullan.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    image_path = Path(args.image)
    output_path = Path(args.output)
    csv_path = Path(args.csv)
    crop_dir = output_path.parent / "crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(image_path)
    detector = PlateDetector(args.weights, confidence=args.conf, image_size=args.imgsz)
    detections = detector.detect(image, max_plates=args.max_plates)

    rows: list[dict[str, str]] = []
    if not detections:
        annotated = draw_plate_result(image, None, "", "tespit_edilemedi")
        save_image(output_path, annotated)
        rows.append(
            _row(
                image_path=image_path,
                index=0,
                text="",
                ocr_confidence=0.0,
                detection_confidence=0.0,
                status="tespit_edilemedi",
                crop_path="",
                output_path=output_path,
            )
        )
        _write_csv(csv_path, rows)
        print("Plaka tespit edilemedi.")
        return 2

    reader = EasyOCRPlateReader(gpu=args.gpu)
    best_detection = detections[0]
    best_text = ""
    best_status = "ocr_okunamadi"

    for index, detection in enumerate(detections, start=1):
        plate_crop = crop_plate(image, detection)
        crop_path = crop_dir / f"{image_path.stem}_plate_{index}.jpg"
        save_image(crop_path, plate_crop)

        variants = build_ocr_variants(plate_crop)
        candidate = reader.read_best(variants)
        status = plate_status(candidate.text, candidate.confidence, args.ocr_conf)
        if index == 1:
            best_text = candidate.text
            best_status = status

        rows.append(
            _row(
                image_path=image_path,
                index=index,
                text=candidate.text,
                ocr_confidence=candidate.confidence,
                detection_confidence=detection.confidence,
                status=status,
                crop_path=crop_path,
                output_path=output_path,
            )
        )

    annotated = draw_plate_result(image, best_detection, best_text, best_status)
    save_image(output_path, annotated)
    _write_csv(csv_path, rows)

    print(f"Islenen gorsel: {image_path}")
    print(f"Okunan plaka: {best_text or 'OCR okunamadi'}")
    print(f"Durum: {best_status}")
    print(f"Cikti gorseli: {output_path}")
    print(f"CSV: {csv_path}")
    return 0 if best_status == "basarili" else 1


def _row(
    image_path: Path,
    index: int,
    text: str,
    ocr_confidence: float,
    detection_confidence: float,
    status: str,
    crop_path: Path | str,
    output_path: Path,
) -> dict[str, str]:
    return {
        "image_path": str(image_path),
        "plate_index": str(index),
        "detected_text": text,
        "ocr_confidence": f"{ocr_confidence:.4f}",
        "detection_confidence": f"{detection_confidence:.4f}",
        "status": status,
        "crop_path": str(crop_path),
        "output_path": str(output_path),
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())

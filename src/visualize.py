from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    from .detector import PlateDetection
except ImportError:
    from detector import PlateDetection


def draw_plate_result(
    image: np.ndarray,
    detection: PlateDetection | None,
    plate_text: str,
    status: str,
) -> np.ndarray:
    canvas = image.copy()
    if detection is None:
        _draw_banner(canvas, "Plaka tespit edilemedi", (0, 0, 255))
        return canvas

    x1, y1, x2, y2 = detection.xyxy
    color = (0, 180, 0) if status == "basarili" else (0, 180, 255)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

    label = plate_text if plate_text else "OCR okunamadi"
    label = f"{label} | det:{detection.confidence:.2f}"
    text_y = max(y1 - 10, 20)
    cv2.putText(canvas, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return canvas


def save_image(path: str | Path, image: np.ndarray) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), image)
    if not ok:
        raise IOError(f"Gorsel kaydedilemedi: {output_path}")
    return output_path


def _draw_banner(image: np.ndarray, message: str, color: tuple[int, int, int]) -> None:
    cv2.rectangle(image, (0, 0), (image.shape[1], 44), (255, 255, 255), -1)
    cv2.putText(image, message, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

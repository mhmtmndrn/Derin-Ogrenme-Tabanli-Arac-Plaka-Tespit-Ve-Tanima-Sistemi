from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class PlateDetection:
    """Single license plate detection in pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    class_id: int
    class_name: str

    @property
    def xyxy(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class PlateDetector:
    """YOLO wrapper for plate detection."""

    def __init__(self, weights_path: str | Path, confidence: float = 0.25, image_size: int = 640) -> None:
        self.weights_path = Path(weights_path)
        if not self.weights_path.exists():
            raise FileNotFoundError(
                f"Model agirligi bulunamadi: {self.weights_path}. "
                "Kaggle egitiminden sonra best.pt dosyasini models/plate_detector.pt olarak kaydedin."
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("ultralytics paketi kurulu degil. `pip install -r requirements.txt` calistirin.") from exc

        self.model = YOLO(str(self.weights_path))
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, image: np.ndarray, max_plates: int = 1) -> list[PlateDetection]:
        """Detect license plates and return the highest-confidence boxes."""
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            imgsz=self.image_size,
            verbose=False,
            max_det=max(max_plates, 1),
        )
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        names = self._names()
        detections: list[PlateDetection] = []
        for box in result.boxes:
            xyxy = box.xyxy[0].detach().cpu().numpy().astype(int)
            conf = float(box.conf[0].detach().cpu().item())
            class_id = int(box.cls[0].detach().cpu().item())
            class_name = names.get(class_id, str(class_id))
            detections.append(
                PlateDetection(
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    confidence=conf,
                    class_id=class_id,
                    class_name=class_name,
                )
            )

        detections.sort(key=lambda item: item.confidence, reverse=True)
        return detections[:max_plates]

    def _names(self) -> dict[int, str]:
        raw_names = getattr(self.model, "names", {}) or {}
        if isinstance(raw_names, dict):
            return {int(key): str(value) for key, value in raw_names.items()}
        if isinstance(raw_names, Iterable):
            return {index: str(value) for index, value in enumerate(raw_names)}
        return {}


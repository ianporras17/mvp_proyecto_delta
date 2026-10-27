"""LPD-YuNet FP32: BGR original, cuadriláteros y cajas T024, sin resize ni MPC."""

from dataclasses import dataclass
import hashlib
import math
from numbers import Integral
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from client.regions import DetectedRegion, FrameDetections, RegionType


MODEL_FILENAME = "license_plate_detection_lpd_yunet_2023mar.onnx"
MODEL_SHA256 = "6d4978a7b6d25514d5e24811b82bfb511d166bdd8ca3b03aa63c1623d4d039c7"
MODEL_REVISION = "962e8d8849878106cb84039dab61cf97bb683b31"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / MODEL_FILENAME
DEFAULT_PLATE_CONFIDENCE_THRESHOLD = 0.6


class PlateDetectionError(RuntimeError):
    """Carga, entrada o inferencia inválida; no equivale a cero placas."""


@dataclass(frozen=True)
class PlateDetection:
    """Cuatro esquinas originales y su envolvente recortada a la imagen."""

    quadrilateral: tuple[tuple[float, float], ...]
    confidence: float
    region: DetectedRegion

    def __post_init__(self) -> None:
        try:
            corners = tuple(tuple(float(value) for value in point) for point in self.quadrilateral)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError("quadrilateral requiere cuatro puntos finitos (x, y)") from exc
        if len(corners) != 4 or any(
            len(point) != 2 or not all(math.isfinite(value) for value in point)
            for point in corners
        ):
            raise ValueError("quadrilateral requiere cuatro puntos finitos (x, y)")
        if not math.isfinite(self.confidence) or not 0 <= self.confidence <= 1:
            raise ValueError("confidence debe estar entre 0 y 1")
        if not isinstance(self.region, DetectedRegion) or self.region.region_type != RegionType.PLATE:
            raise ValueError("region debe ser DetectedRegion de tipo PLATE")
        object.__setattr__(self, "quadrilateral", corners)
        object.__setattr__(self, "confidence", float(self.confidence))


def to_frame_detections(
    frame_index: int, detections: Iterable[PlateDetection]
) -> FrameDetections:
    """Adapta al índice de la secuencia realmente procesada, nunca al paquete."""
    results = tuple(detections)
    if any(not isinstance(detection, PlateDetection) for detection in results):
        raise ValueError("detections debe contener PlateDetection")
    return FrameDetections.success(frame_index, (detection.region for detection in results))


class LPDYuNet:
    """Inferencia OpenCV DNN a tamaño nativo, según el preprocesamiento oficial.

    El modelo produce loc/conf/iou, no cajas xywh listas. Se decodifican las
    cuatro esquinas de loc y NMS utiliza sus envolventes xywh. El umbral oficial
    es 0.8; el perfil del proyecto usa 0.6. La confianza no es precisión medida.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        *,
        confidence_threshold: float = DEFAULT_PLATE_CONFIDENCE_THRESHOLD,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        keep_top_k: int = 750,
    ) -> None:
        for name, value in (
            ("confidence_threshold", confidence_threshold), ("nms_threshold", nms_threshold)
        ):
            if isinstance(value, bool) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} debe estar entre 0 y 1")
        for name, value in (("top_k", top_k), ("keep_top_k", keep_top_k)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
                raise ValueError(f"{name} debe ser un entero positivo")
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MODEL_PATH
        self.confidence_threshold = float(confidence_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.keep_top_k = int(keep_top_k)
        self._input_size: tuple[int, int] | None = None
        self._priors = np.empty((0, 4), dtype=np.float32)
        try:
            if not self.model_path.is_file():
                raise PlateDetectionError(f"No se encontró el modelo LPD-YuNet: {self.model_path}")
            if model_path is None:
                digest = hashlib.sha256(self.model_path.read_bytes()).hexdigest()
                if digest != MODEL_SHA256:
                    raise PlateDetectionError("El SHA-256 del modelo LPD-YuNet no coincide")
            self._model = cv2.dnn.readNet(str(self.model_path))
            self._model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._model.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except (cv2.error, OSError) as exc:
            raise PlateDetectionError(f"No se pudo cargar LPD-YuNet: {self.model_path}") from exc

    @staticmethod
    def _generate_priors(width: int, height: int) -> np.ndarray:
        # Exactamente los mapas y anclas de lpd_yunet.py oficial. Las divisiones
        # son enteras; no asumir ceil/8 para tamaños nativos impares.
        rows, columns = ((height + 1) // 2) // 2, ((width + 1) // 2) // 2
        priors = []
        for step, sizes in zip((8, 16, 32, 64), ((10, 16, 24), (32, 48), (64, 96), (128, 192, 256))):
            rows, columns = rows // 2, columns // 2
            for row in range(rows):
                for column in range(columns):
                    for size in sizes:
                        priors.append(((column + 0.5) * step / width,
                                       (row + 0.5) * step / height,
                                       size / width, size / height))
        return np.asarray(priors, dtype=np.float32).reshape(-1, 4)

    def detect(self, image: np.ndarray) -> tuple[PlateDetection, ...]:
        """Devuelve detecciones o éxito vacío; un error técnico lanza excepción."""
        if (
            not isinstance(image, np.ndarray) or image.dtype != np.uint8
            or image.ndim != 3 or image.shape[2] != 3
            or image.shape[0] < 8 or image.shape[1] < 8
        ):
            raise PlateDetectionError("LPD-YuNet requiere una imagen BGR uint8 HxWx3 de al menos 8x8")
        height, width = image.shape[:2]
        try:
            if self._input_size != (width, height):
                self._priors = self._generate_priors(width, height)
                self._input_size = (width, height)
            # Sin resize, normalización, intercambio de canales ni crop.
            self._model.setInput(cv2.dnn.blobFromImage(image))
            outputs = self._model.forward(["loc", "conf", "iou"])
            return self._decode(outputs, width, height)
        except PlateDetectionError:
            raise
        except (cv2.error, TypeError, ValueError, OverflowError, FloatingPointError, MemoryError) as exc:
            raise PlateDetectionError(f"Falló LPD-YuNet para una imagen {width}x{height}") from exc

    def detect_frame(self, image: np.ndarray, frame_index: int) -> FrameDetections:
        """Adaptador T024 que conserva explícitamente los fallos del detector."""
        FrameDetections.success(frame_index)  # Validar el índice antes de inferir.
        try:
            return to_frame_detections(frame_index, self.detect(image))
        except PlateDetectionError as exc:
            return FrameDetections.failure(frame_index, str(exc))

    def _decode(self, outputs: Iterable[np.ndarray], width: int, height: int) -> tuple[PlateDetection, ...]:
        arrays = tuple(np.asarray(output) for output in outputs)
        if len(arrays) != 3:
            raise PlateDetectionError("LPD-YuNet debe devolver loc, conf e iou")
        arrays = tuple(array[0] if array.ndim == 3 and array.shape[0] == 1 else array for array in arrays)
        expected = ((len(self._priors), 14), (len(self._priors), 2), (len(self._priors), 1))
        if any(array.shape != shape for array, shape in zip(arrays, expected)):
            raise PlateDetectionError("Las salidas LPD-YuNet no coinciden con las anclas y dimensiones nativas")
        if any(not np.isfinite(array).all() for array in arrays):
            raise PlateDetectionError("LPD-YuNet devolvió valores no finitos")
        loc, conf, iou = arrays
        if (conf < 0).any() or (conf > 1).any():
            raise PlateDetectionError("LPD-YuNet devolvió probabilidades de clase inválidas")
        scores = np.sqrt(conf[:, 1] * np.clip(iou[:, 0], 0, 1))
        candidates = np.flatnonzero(scores > self.confidence_threshold)
        if not len(candidates):
            return ()
        priors, offsets = self._priors[candidates], loc[candidates]
        corners = np.stack([
            (priors[:, :2] + offsets[:, start:start + 2] * 0.1 * priors[:, 2:4])
            * np.array((width, height), dtype=np.float32)
            for start in (4, 6, 10, 12)
        ], axis=1)
        if not np.isfinite(corners).all():
            raise PlateDetectionError("LPD-YuNet produjo esquinas no finitas")
        detections = []
        for corner, score in zip(corners, scores[candidates]):
            minimum, maximum = corner.min(axis=0), corner.max(axis=0)
            if (maximum <= minimum).any():
                raise PlateDetectionError("LPD-YuNet produjo una placa de geometría inválida")
            left, top = max(0.0, float(minimum[0])), max(0.0, float(minimum[1]))
            right, bottom = min(float(width), float(maximum[0])), min(float(height), float(maximum[1]))
            if right <= left or bottom <= top:
                continue  # Caja bien formada pero completamente fuera del cuadro.
            detections.append(PlateDetection(
                tuple(tuple(float(value) for value in point) for point in corner),
                float(score), DetectedRegion(RegionType.PLATE, left, top, right - left, bottom - top),
            ))
        if not detections:
            return ()
        # NMS requiere x/y/ancho/alto, no los primeros dos puntos del cuadrilátero.
        indices = cv2.dnn.NMSBoxes(
            [(d.region.x, d.region.y, d.region.width, d.region.height) for d in detections],
            [d.confidence for d in detections], self.confidence_threshold,
            self.nms_threshold, top_k=self.top_k,
        )
        return tuple(detections[int(index)] for index in np.asarray(indices).reshape(-1)[:self.keep_top_k])

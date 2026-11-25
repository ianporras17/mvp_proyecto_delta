"""YOLOX-S COCO local y conteo por trayectorias, separado de las máscaras MPC.

Modelo oficial OpenCV Zoo (Apache-2.0). El adaptador reproduce RGB 0..255,
letterbox superior/izquierdo 640x640 y decodificación YOLOX de su referencia.
No identifica personas: IDs temporales por video, con pérdida posible en reingresos.
"""

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

import cv2
import numpy as np


DEFAULT_YOLO_MODEL = Path(__file__).with_name("models") / "object_detection_yolox_2022nov.onnx"
MODEL_SHA256 = "c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063"
CLASSES = {0: "person", 2: "car"}


class CountingError(RuntimeError):
    """La inferencia o seguimiento no pudo completarse; no es un conteo cero."""


@dataclass(frozen=True)
class ObjectDetection:
    kind: str
    box: tuple[float, float, float, float]  # xywh en la imagen original
    confidence: float

    def __post_init__(self):
        if self.kind not in CLASSES.values() or len(self.box) != 4:
            raise ValueError("Clase/caja de conteo inválida")
        if not all(math.isfinite(v) for v in (*self.box, self.confidence)):
            raise ValueError("La detección debe ser finita")
        if min(self.box[2:]) <= 0 or not 0 <= self.confidence <= 1:
            raise ValueError("Geometría o confianza inválida")


class YOLODetector:
    def __init__(self, model_path=DEFAULT_YOLO_MODEL, confidence_threshold=0.4):
        if not 0 < confidence_threshold < 1:
            raise ValueError("El umbral YOLO debe estar entre 0 y 1")
        if not Path(model_path).is_file():
            raise CountingError(f"No se encontró el modelo YOLOX: {model_path}")
        if hashlib.sha256(Path(model_path).read_bytes()).hexdigest() != MODEL_SHA256:
            raise CountingError("El modelo YOLOX no coincide con el recurso oficial verificado")
        self.threshold = confidence_threshold
        try:
            self.net = cv2.dnn.readNetFromONNX(str(model_path))
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        except cv2.error as exc:
            raise CountingError("No se pudo cargar YOLOX") from exc
        grids, strides = [], []
        for stride in (8, 16, 32):
            y, x = np.mgrid[:640 // stride, :640 // stride]
            grids.append(np.stack((x, y), axis=-1).reshape(-1, 2))
            strides.extend([stride] * x.size)
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.asarray(strides, dtype=np.float32)[:, None]

    def detect(self, original_bgr: np.ndarray) -> tuple[ObjectDetection, ...]:
        if original_bgr.ndim != 3 or original_bgr.shape[2] != 3 or original_bgr.dtype != np.uint8 or not original_bgr.size:
            raise CountingError("YOLO requiere una imagen BGR uint8 de tres canales")
        height, width = original_bgr.shape[:2]
        ratio = min(640 / width, 640 / height)
        resized = (max(1, int(width * ratio)), max(1, int(height * ratio)))
        image = np.full((640, 640, 3), 114, dtype=np.float32)
        try:
            image[:resized[1], :resized[0]] = cv2.resize(
                cv2.cvtColor(original_bgr, cv2.COLOR_BGR2RGB), resized)
            self.net.setInput(np.ascontiguousarray(image.transpose(2, 0, 1)[None]))
            output = self.net.forward()
            return self._decode(output, ratio, width, height)
        except (cv2.error, ValueError, OverflowError) as exc:
            raise CountingError(f"Falló la inferencia YOLOX: {exc}") from exc

    def _decode(self, output, ratio, width, height):
        if output.shape != (1, 8400, 85) or not np.isfinite(output).all():
            raise CountingError("YOLOX devolvió una salida incompatible/no finita")
        rows = output[0]
        class_ids = rows[:, 5:].argmax(axis=1)
        scores = rows[:, 4] * rows[np.arange(len(rows)), class_ids + 5]
        selected = np.flatnonzero(np.isin(class_ids, tuple(CLASSES)) & (scores >= self.threshold))
        detections = []
        for class_id, kind in CLASSES.items():
            indices = selected[class_ids[selected] == class_id]
            boxes, confidences = [], []
            for index in indices:
                center = (rows[index, :2] + self.grid[index]) * self.strides[index]
                size = np.exp(np.clip(rows[index, 2:4], -20, 20)) * self.strides[index]
                left, top = np.maximum(0, (center - size / 2) / ratio)
                right, bottom = np.minimum((width, height), (center + size / 2) / ratio)
                if right > left and bottom > top:
                    boxes.append([float(left), float(top), float(right-left), float(bottom-top)])
                    confidences.append(float(scores[index]))
            keep = cv2.dnn.NMSBoxes(boxes, confidences, self.threshold, 0.45)
            for index in np.asarray(keep, dtype=int).reshape(-1):
                detections.append(ObjectDetection(kind, tuple(boxes[index]), confidences[index]))
        return tuple(detections)


def _iou(first, second):
    x, y = max(first[0], second[0]), max(first[1], second[1])
    right = min(first[0]+first[2], second[0]+second[2])
    bottom = min(first[1]+first[3], second[1]+second[3])
    intersection = max(0, right-x) * max(0, bottom-y)
    return intersection / (first[2]*first[3] + second[2]*second[3] - intersection)


@dataclass
class _Track:
    kind: str
    box: tuple
    last_frame: int
    hits: int = 1
    counted: bool = False
    velocity: tuple = (0., 0.)


class VideoCounter:
    """Asociación uno-a-uno por clase, IoU y velocidad; memoria acotada por TTL.

    Se cuentan trayectorias confirmadas, no observaciones por frame. No hay
    reconocimiento biométrico ni deduplicación garantizada tras salir de escena.
    """

    def __init__(self, detector=None, *, max_missed=60, min_hits=3):
        if max_missed < 0 or min_hits < 1:
            raise ValueError("Parámetros de seguimiento inválidos")
        self.detector = detector if detector is not None else YOLODetector()
        self.max_missed, self.min_hits = max_missed, min_hits
        self.tracks = {}
        self.next_id = 1
        self.frame_index = -1
        self.totals = {"person": 0, "car": 0}

    def observe(self, original_bgr, frame_index):
        self.update(self.detector.detect(original_bgr), frame_index)

    def update(self, detections, frame_index):
        if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index != self.frame_index + 1:
            raise CountingError("El conteo requiere índices consecutivos desde cero")
        self.frame_index = frame_index
        self.tracks = {key: track for key, track in self.tracks.items()
                       if frame_index - track.last_frame <= self.max_missed + 1}
        candidates = []
        for key, track in self.tracks.items():
            gap = frame_index - track.last_frame
            predicted = (track.box[0]+track.velocity[0]*gap,
                         track.box[1]+track.velocity[1]*gap, *track.box[2:])
            for index, detection in enumerate(detections):
                if track.kind == detection.kind:
                    score = max(_iou(predicted, detection.box), _iou(track.box, detection.box))
                    if score >= 0.3:
                        candidates.append((score, key, index))
        matched_tracks, matched_detections = set(), set()
        for _, key, index in sorted(candidates, reverse=True):
            if key in matched_tracks or index in matched_detections:
                continue
            track, detection = self.tracks[key], detections[index]
            gap = frame_index - track.last_frame
            track.velocity = tuple((detection.box[i]-track.box[i])/gap for i in (0, 1))
            track.box, track.last_frame = detection.box, frame_index
            track.hits += 1
            matched_tracks.add(key)
            matched_detections.add(index)
        for index, detection in enumerate(detections):
            if index not in matched_detections:
                self.tracks[self.next_id] = _Track(detection.kind, detection.box, frame_index)
                self.next_id += 1
        for track in self.tracks.values():
            if not track.counted and track.hits >= self.min_hits:
                track.counted = True
                self.totals[track.kind] += 1

    def summary(self):
        return {"people": self.totals["person"], "cars": self.totals["car"],
                "frames_analyzed": self.frame_index + 1,
                "method": "YOLOX-S + IoU/motion tracking", "min_hits": self.min_hits,
                "limitation": "Trayectorias estimadas; oclusiones, cruces o reingresos pueden omitir o duplicar objetos."}

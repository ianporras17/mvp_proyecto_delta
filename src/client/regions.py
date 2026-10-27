"""Contrato T024 de regiones públicas; no detecta, anonimiza ni transmite píxeles."""

from dataclasses import dataclass
from enum import Enum
from fractions import Fraction
import math
from numbers import Integral, Real
from typing import Iterable


class RegionType(str, Enum):
    FACE = "face"
    PLATE = "plate"


class DetectorExecutionError(RuntimeError):
    """La preparación no puede continuar después de un fallo del detector."""


def _integer(value: int, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} debe ser un entero mayor o igual a {minimum}")
    return int(value)


def _region_type(value: RegionType) -> RegionType:
    try:
        return RegionType(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("region_type debe ser face o plate") from exc


def _dimensions(size: tuple[int, int], name: str) -> tuple[int, int]:
    try:
        width, height = size
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} requiere (ancho, alto)") from exc
    width = _integer(width, f"{name}.width", 1)
    height = _integer(height, f"{name}.height", 1)
    try:
        if not math.isfinite(width) or not math.isfinite(height):
            raise ValueError(f"{name} debe tener dimensiones finitas")
    except OverflowError as exc:
        raise ValueError(f"{name} debe tener dimensiones finitas") from exc
    return width, height


@dataclass(frozen=True)
class DetectedRegion:
    """Caja original (x, y, ancho, alto); admite posiciones fuera de la imagen."""

    region_type: RegionType
    x: float
    y: float
    width: float
    height: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_type", _region_type(self.region_type))
        for name in ("x", "y", "width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError(f"{name} debe ser un número finito")
            try:
                value = float(value)
            except (OverflowError, ValueError) as exc:
                raise ValueError(f"{name} debe ser un número finito") from exc
            if not math.isfinite(value):
                raise ValueError(f"{name} debe ser un número finito")
            object.__setattr__(self, name, value)
        if self.width <= 0 or self.height <= 0:
            raise ValueError("width y height deben ser positivos")
        if not math.isfinite(self.x + self.width) or not math.isfinite(self.y + self.height):
            raise ValueError("Los límites finales de la caja deben ser finitos")


@dataclass(frozen=True)
class SensitiveRegion:
    """Caja procesada entera: [x, x+width) × [y, y+height), frame_index desde 0."""

    frame_index: int
    region_type: RegionType
    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "region_type", _region_type(self.region_type))
        for name in ("frame_index", "x", "y", "width", "height"):
            minimum = 1 if name in ("width", "height") else 0
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum))


@dataclass(frozen=True)
class FrameDetections:
    """Resultado de un fotograma procesado: error=None significa detección exitosa."""

    frame_index: int
    detections: tuple[DetectedRegion, ...] = ()
    error: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "frame_index", _integer(self.frame_index, "frame_index"))
        try:
            detections = tuple(self.detections)
        except TypeError as exc:
            raise ValueError("detections debe ser una colección de DetectedRegion") from exc
        if any(not isinstance(region, DetectedRegion) for region in detections):
            raise ValueError("detections solo admite DetectedRegion")
        object.__setattr__(self, "detections", detections)
        if self.error is not None:
            if not isinstance(self.error, str) or not self.error.strip():
                raise ValueError("Un fallo requiere un detalle de error no vacío")
            if detections:
                raise ValueError("Un detector fallido no puede declarar detecciones exitosas")
            object.__setattr__(self, "error", self.error.strip())

    @classmethod
    def success(
        cls, frame_index: int, detections: Iterable[DetectedRegion] = ()
    ) -> "FrameDetections":
        try:
            detections = tuple(detections)
        except TypeError as exc:
            raise ValueError("detections debe ser una colección de DetectedRegion") from exc
        return cls(frame_index, detections)

    @classmethod
    def failure(cls, frame_index: int, error: str) -> "FrameDetections":
        if error is None:
            raise ValueError("Un fallo requiere un detalle de error no vacío")
        return cls(frame_index, error=error)

    @property
    def succeeded(self) -> bool:
        return self.error is None


def transform_region(
    detection: DetectedRegion,
    *,
    frame_index: int,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> SensitiveRegion | None:
    """Recorta y escala; floor en inicios, ceil en finales. Exterior válido -> None."""
    frame_index = _integer(frame_index, "frame_index")
    source_width, source_height = _dimensions(source_size, "source_size")
    target_width, target_height = _dimensions(target_size, "target_size")
    if not isinstance(detection, DetectedRegion):
        raise ValueError("detection debe ser DetectedRegion")
    # Preservar los valores float recibidos sin perder cajas subpíxel al sumar
    # sus extremos ni redondear hacia dentro por precisión de la escala.
    source_x, source_y = Fraction(detection.x), Fraction(detection.y)
    left = max(0, source_x)
    top = max(0, source_y)
    right = min(source_width, source_x + Fraction(detection.width))
    bottom = min(source_height, source_y + Fraction(detection.height))
    if right <= left or bottom <= top:
        return None

    scale_x = Fraction(target_width, source_width)
    scale_y = Fraction(target_height, source_height)
    x = max(0, min(target_width, math.floor(left * scale_x)))
    y = max(0, min(target_height, math.floor(top * scale_y)))
    end_x = max(0, min(target_width, math.ceil(right * scale_x)))
    end_y = max(0, min(target_height, math.ceil(bottom * scale_y)))
    return SensitiveRegion(frame_index, detection.region_type, x, y, end_x - x, end_y - y)


def prepare_frame_regions(
    frame: FrameDetections,
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[SensitiveRegion, ...]:
    """Prepara un fotograma; un fallo nunca se convierte en una colección vacía."""
    if not isinstance(frame, FrameDetections):
        raise ValueError("frame debe ser FrameDetections")
    if not frame.succeeded:
        raise DetectorExecutionError(f"Detector fallido en frame_index={frame.frame_index}: {frame.error}")
    source_size = _dimensions(source_size, "source_size")
    target_size = _dimensions(target_size, "target_size")
    regions = []
    for detection in frame.detections:
        region = transform_region(
            detection, frame_index=frame.frame_index,
            source_size=source_size, target_size=target_size,
        )
        if region is not None:
            regions.append(region)
    return tuple(regions)


def prepare_regions(
    frames: Iterable[FrameDetections],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    frame_count: int | None = None,
) -> tuple[SensitiveRegion, ...]:
    """Secuencia completa 0..n-1, incluidos vacíos; no entrega resultados parciales.

    Si se conoce la cantidad real de cuadros, pasar frame_count para rechazar
    resultados faltantes o adicionales. No transmite datos ni altera los jobs.
    """
    source_size = _dimensions(source_size, "source_size")
    target_size = _dimensions(target_size, "target_size")
    if frame_count is not None:
        frame_count = _integer(frame_count, "frame_count", 1)
    regions = []
    count = 0
    for index, frame in enumerate(frames):
        if not isinstance(frame, FrameDetections) or frame.frame_index != index:
            raise ValueError("frame_index debe seguir la secuencia procesada 0, 1, 2, ...")
        if frame_count is not None and index >= frame_count:
            raise ValueError("Hay más resultados de detección que fotogramas procesados")
        regions.extend(prepare_frame_regions(frame, source_size=source_size, target_size=target_size))
        count += 1
    if frame_count is not None and count != frame_count:
        raise ValueError("Faltan resultados de detección para fotogramas procesados")
    return tuple(regions)


def logical_batch_index(frame_index: int, *, batch_size: int = 10) -> int:
    """Ordinal lógico del paquete; independiente de FPS y de bloques ExternalIO."""
    return _integer(frame_index, "frame_index") // _integer(batch_size, "batch_size", 1)

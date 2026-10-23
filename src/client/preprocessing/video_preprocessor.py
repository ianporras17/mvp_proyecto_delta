"""Preprocesamiento local de video a pixels uint8 para el MPC."""

from dataclasses import asdict, dataclass
from collections.abc import Callable
import math
from pathlib import Path

import cv2
import numpy as np

from client.preprocessing.yunet import YuNet
from mpc.video_profile import (
    DEFAULT_MPC_FPS,
    MAX_CONFIGURED_FPS,
    MAX_VIDEO_DURATION_SECONDS,
    MpcVideoProfile,
)


class VideoPreprocessingError(RuntimeError):
    """El MP4 no cumple el perfil o no pudo decodificarse."""


FACE_CONFIDENCE_THRESHOLD = 0.6


@dataclass(frozen=True)
class FaceRegion:
    """Caja de un rostro en las coordenadas del frame incluido en el payload."""

    frame_id: int
    x: int
    y: int
    width: int
    height: int
    confidence: float

    def __post_init__(self) -> None:
        if isinstance(self.frame_id, bool) or not isinstance(self.frame_id, int) or self.frame_id < 0:
            raise ValueError("frame_id debe ser un entero no negativo")
        for name in ("x", "y"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} debe ser un entero no negativo")
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} debe ser un entero positivo")
        if not math.isfinite(self.confidence):
            raise ValueError("confidence debe ser finita")


def scale_bounding_box(
    box: tuple[float, float, float, float],
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Escala (x, y, ancho, alto) utilizando tamaños reales (ancho, alto).

    Es geometría local para un resize sin recorte ni padding, no un contrato
    de regiones. No redondea ni recorta: T024 podrá decidir esa política.
    """
    if len(box) != 4 or len(source_size) != 2 or len(target_size) != 2:
        raise ValueError("Se requieren una caja de cuatro valores y dos tamaños")
    dimensions = (*source_size, *target_size)
    if any(not math.isfinite(value) or value <= 0 for value in dimensions):
        raise ValueError("Las dimensiones deben ser positivas y finitas")
    if any(not math.isfinite(value) for value in box) or box[2] < 0 or box[3] < 0:
        raise ValueError("La caja debe ser finita y no tener tamaños negativos")
    scale_x = target_size[0] / source_size[0]
    scale_y = target_size[1] / source_size[1]
    x, y, width, height = box
    return x * scale_x, y * scale_y, width * scale_x, height * scale_y


@dataclass(frozen=True)
class VideoMetadata:
    frame_count: int
    height: int
    width: int
    channels: int
    fps: float
    dtype: str
    version: int = 2

    def as_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


@dataclass(frozen=True)
class PreprocessedVideo:
    """Pixels grayscale contiguos en orden frame/fila/columna."""

    payload: bytes
    metadata: VideoMetadata
    source_size: tuple[int, int] | None = None
    regions: tuple[FaceRegion, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "regions", tuple(self.regions))


class VideoPreprocessor:
    """Reduce resolución/FPS y convierte a grayscale exclusivamente en cliente."""



    def __init__(
        self,
        *,
        max_duration_seconds: float = MAX_VIDEO_DURATION_SECONDS,
        max_width: int | None = None,
        max_height: int | None = None,
        target_fps: float = DEFAULT_MPC_FPS,
    ) -> None:
        if (
            not math.isfinite(max_duration_seconds)
            or not 0 < max_duration_seconds <= MAX_VIDEO_DURATION_SECONDS
        ):
            raise ValueError(
                f"max_duration_seconds debe ser positivo y no superar {MAX_VIDEO_DURATION_SECONDS}"
            )
        if not math.isfinite(target_fps) or not 0 < target_fps <= MAX_CONFIGURED_FPS:
            raise ValueError(f"target_fps debe ser positivo y no superar {MAX_CONFIGURED_FPS}")
        profile = MpcVideoProfile.from_environment(max_width, max_height)
        self._max_duration = float(max_duration_seconds)
        self._max_width = profile.max_width
        self._max_height = profile.max_height
        self._target_fps = float(target_fps)

    @staticmethod
    def _face_regions(results: np.ndarray, frame_id: int,
                      source_size: tuple[int, int], target_size: tuple[int, int]) -> list[FaceRegion]:
        """Convierte cajas YuNet a límites enteros recortados del frame final."""
        if results is None:
            return []
        source_width, source_height = source_size
        target_width, target_height = target_size
        scale_x = target_width / source_width
        scale_y = target_height / source_height
        # OpenCV devuelve puntuaciones FP32: comparar en esa misma precisión
        # evita aceptar 0.60 como 0.6000000238 tras convertirlo a float de Python.
        confidence_threshold = float(np.float32(FACE_CONFIDENCE_THRESHOLD))
        regions = []
        for detection in results:
            if len(detection) < 5:
                raise VideoPreprocessingError("YuNet devolvió una detección facial incompleta")
            x, y, width, height = (float(value) for value in detection[:4])
            confidence = float(detection[-1])
            if not all(math.isfinite(value) for value in (x, y, width, height, confidence)) or width <= 0 or height <= 0:
                raise VideoPreprocessingError("YuNet devolvió una caja facial inválida")
            if confidence <= confidence_threshold:
                continue
            # Floor el borde superior/izquierdo y ceil el inferior/derecho
            # para conservar la cobertura tras el resize.
            left = max(0, min(target_width, math.floor(x * scale_x)))
            top = max(0, min(target_height, math.floor(y * scale_y)))
            right = max(0, min(target_width, math.ceil((x + width) * scale_x)))
            bottom = max(0, min(target_height, math.ceil((y + height) * scale_y)))
            if right > left and bottom > top:
                regions.append(FaceRegion(frame_id, left, top, right - left, bottom - top, confidence))
        return regions

    def process(
        self,
        video_path: str | Path,
        model: YuNet,
        *,
        on_sample: Callable[[np.ndarray, int], None] | None = None,
    ) -> PreprocessedVideo:
        """Prepara los cuadros y permite observar exactamente la misma selección.

        ``on_sample`` recibe una copia BGR original y el índice desde cero de
        la secuencia procesada, antes de gris/resize. Un error del callback
        aborta la preparación: nunca se interpreta como detección vacía.
        """
        path = Path(video_path)
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            raise VideoPreprocessingError(f"OpenCV no pudo abrir el MP4: {path}")

        source_fps = float(capture.get(cv2.CAP_PROP_FPS))
        reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not math.isfinite(source_fps) or source_fps <= 0:
            capture.release()
            raise VideoPreprocessingError("El MP4 no declara un FPS válido")
        if reported_frames > 0 and reported_frames / source_fps > self._max_duration + 1e-6:
            capture.release()
            raise VideoPreprocessingError(
                f"El video excede la duración máxima de {self._max_duration:g} segundos"
            )

        output_fps = min(source_fps, self._target_fps)
        next_sample_time = 0.0
        decoded_index = 0
        target_size: tuple[int, int] | None = None
        source_size: tuple[int, int] | None = None
        frame_bytes: list[bytes] = []
        regions: list[FaceRegion] = []

        try:
            while True:
                success, frame = capture.read()
                if not success:
                    break
                timestamp = decoded_index / source_fps
                decoded_index += 1
                if timestamp >= self._max_duration:
                    raise VideoPreprocessingError(
                        f"El video excede la duración máxima de {self._max_duration:g} segundos"
                    )

                if timestamp + 1e-9 < next_sample_time:
                    continue

                original_size = (frame.shape[1], frame.shape[0])
                if source_size is None:
                    source_size = original_size
                if (frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8):
                    raise VideoPreprocessingError("YuNet requiere cuadros BGR originales uint8")
                if original_size != source_size:
                    raise VideoPreprocessingError("Las dimensiones originales cambian entre cuadros seleccionados")
                if on_sample is not None:
                    on_sample(frame.copy(), len(frame_bytes))
                model.setInputSize([frame.shape[1], frame.shape[0]])
                detections = model.infer(frame)

                if target_size is None:
                    target_size = self._scaled_size(frame.shape[1], frame.shape[0])
                regions.extend(self._face_regions(
                    detections, len(frame_bytes), original_size, target_size
                ))
                gray = self._to_grayscale(frame)
                if (gray.shape[1], gray.shape[0]) != target_size:
                    gray = cv2.resize(gray, target_size, interpolation=cv2.INTER_AREA)
                gray = np.ascontiguousarray(gray, dtype=np.uint8)
                frame_bytes.append(gray.tobytes())
                next_sample_time += 1.0 / output_fps
        except VideoPreprocessingError:
            raise
        except (cv2.error, MemoryError, OverflowError, ValueError) as exc:
            raise VideoPreprocessingError(
                "No se pudo convertir el video a pixels grayscale"
            ) from exc
        finally:
            capture.release()

        if not frame_bytes or target_size is None:
            raise VideoPreprocessingError("El MP4 no contiene frames decodificables")

        width, height = target_size
        max_frames = math.ceil(self._max_duration * output_fps)
        if len(frame_bytes) > max_frames:
            raise VideoPreprocessingError("El muestreo produjo más frames de los permitidos")
        metadata = VideoMetadata(
            frame_count=len(frame_bytes),
            height=height,
            width=width,
            channels=1,
            fps=output_fps,
            dtype="uint8",
            version=2,
        )
        payload = b"".join(frame_bytes)
        expected_size = metadata.frame_count * metadata.height * metadata.width
        if len(payload) != expected_size:
            raise VideoPreprocessingError("El payload numérico tiene un tamaño inconsistente")
        return PreprocessedVideo(
            payload=payload, metadata=metadata, source_size=source_size, regions=tuple(regions)
        )

    @staticmethod
    def _to_grayscale(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 2:
            return frame
        if frame.ndim != 3:
            raise VideoPreprocessingError("OpenCV devolvió un frame con forma inválida")
        if frame.shape[2] == 1:
            return frame[:, :, 0]
        if frame.shape[2] == 3:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if frame.shape[2] == 4:
            return cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
        raise VideoPreprocessingError("El MP4 usa un número de canales no soportado")

    def _scaled_size(self, width: int, height: int) -> tuple[int, int]:
        scale = min(1.0, self._max_width / width, self._max_height / height)
        # mp4v puede truncar dimensiones impares. Alinear hacia abajo evita
        # ampliar la imagen o reconstruir con un tamaño distinto al anunciado.
        target_width = math.floor(width * scale + 1e-9) // 2 * 2
        target_height = math.floor(height * scale + 1e-9) // 2 * 2
        if target_width < 2 or target_height < 2:
            raise VideoPreprocessingError("El tamaño procesado debe ser al menos 2x2")
        return target_width, target_height

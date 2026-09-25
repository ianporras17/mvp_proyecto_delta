"""Preprocesamiento local de video a pixels uint8 para el MPC."""

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import cv2
import numpy as np


class VideoPreprocessingError(RuntimeError):
    """El MP4 no cumple el perfil o no pudo decodificarse."""


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


class VideoPreprocessor:
    """Reduce resolución/FPS y convierte a grayscale exclusivamente en cliente."""

    def __init__(
        self,
        *,
        max_duration_seconds: float = 20.0,
        max_width: int = 320,
        max_height: int = 240,
        target_fps: float = 5.0,
    ) -> None:
        if max_duration_seconds <= 0:
            raise ValueError("max_duration_seconds debe ser positivo")
        if max_width <= 0 or max_height <= 0:
            raise ValueError("Las dimensiones máximas deben ser positivas")
        if target_fps <= 0:
            raise ValueError("target_fps debe ser positivo")
        self._max_duration = float(max_duration_seconds)
        self._max_width = int(max_width)
        self._max_height = int(max_height)
        self._target_fps = float(target_fps)

    def process(self, video_path: str | Path) -> PreprocessedVideo:
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
        frame_bytes: list[bytes] = []

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

                gray = self._to_grayscale(frame)
                if target_size is None:
                    target_size = self._scaled_size(gray.shape[1], gray.shape[0])
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
        return PreprocessedVideo(payload=payload, metadata=metadata)

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
        target_width = max(1, int(round(width * scale)))
        target_height = max(1, int(round(height * scale)))
        return target_width, target_height

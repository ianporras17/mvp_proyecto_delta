"""Reconstrucción del resultado MPC únicamente en el cliente confiable."""

from pathlib import Path
from uuid import UUID

import cv2
import numpy as np

from client.preprocessing.video_preprocessor import VideoMetadata


class VideoReconstructionError(RuntimeError):
    """El resultado MPC no pudo convertirse a un MP4 reproducible."""


class VideoReconstructor:
    def reconstruct(
        self,
        payload: bytes,
        metadata: VideoMetadata,
        video_id: str,
        output_root: str | Path = "output",
    ) -> Path:
        normalized_id = str(UUID(video_id))
        expected_size = metadata.frame_count * metadata.height * metadata.width
        if len(payload) != expected_size:
            raise VideoReconstructionError(
                f"Resultado MPC inválido: {len(payload)} bytes, se esperaban {expected_size}"
            )

        destination_dir = Path(output_root) / normalized_id
        destination_dir.mkdir(parents=True, exist_ok=True)
        destination = destination_dir / "reconstructed.mp4"
        writer = cv2.VideoWriter(
            str(destination),
            cv2.VideoWriter_fourcc(*"mp4v"),
            metadata.fps,
            (metadata.width, metadata.height),
            isColor=False,
        )
        if not writer.isOpened():
            writer.release()
            raise VideoReconstructionError("OpenCV no pudo crear reconstructed.mp4")

        values = np.frombuffer(payload, dtype=np.uint8)
        frames = values.reshape(
            metadata.frame_count,
            metadata.height,
            metadata.width,
        )
        try:
            for frame in frames:
                writer.write(frame)
        except cv2.error as exc:
            raise VideoReconstructionError("OpenCV no pudo escribir el resultado") from exc
        finally:
            writer.release()

        self._validate_output(destination, metadata)
        return destination

    @staticmethod
    def _validate_output(path: Path, metadata: VideoMetadata) -> None:
        if not path.is_file() or path.stat().st_size == 0:
            raise VideoReconstructionError("El MP4 reconstruido está vacío")
        capture = cv2.VideoCapture(str(path))
        try:
            if not capture.isOpened():
                raise VideoReconstructionError("El MP4 reconstruido no puede abrirse")
            width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if (
                (width, height) != (metadata.width, metadata.height)
                or frames != metadata.frame_count
                or not np.isfinite(fps)
                or abs(fps - metadata.fps) > 0.25
            ):
                raise VideoReconstructionError(
                    "El MP4 reconstruido no conserva dimensiones/frames/FPS válidos"
                )
            decoded_count = 0
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if frame.shape[:2] != (metadata.height, metadata.width):
                    raise VideoReconstructionError("Un frame reconstruido cambió de tamaño")
                decoded_count += 1
            if decoded_count != metadata.frame_count:
                raise VideoReconstructionError("El MP4 no conserva todos los frames")
        finally:
            capture.release()

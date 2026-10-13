"""Límites de resolución y FPS compartidos por el cliente y las APIs del MVP."""

from dataclasses import dataclass
import os


DEFAULT_MPC_WIDTH = 640
MAX_CONFIGURED_WIDTH = 640
MAX_CONFIGURED_HEIGHT = 480
DEFAULT_MPC_FPS = 10.0
MAX_CONFIGURED_FPS = 10
MAX_VIDEO_DURATION_SECONDS = 20
MAX_VIDEO_FRAMES = MAX_VIDEO_DURATION_SECONDS * MAX_CONFIGURED_FPS


@dataclass(frozen=True)
class MpcVideoProfile:
    max_width: int
    max_height: int

    def __post_init__(self) -> None:
        for name, value, ceiling in (
            ("MPC_MAX_WIDTH", self.max_width, MAX_CONFIGURED_WIDTH),
            ("MPC_MAX_HEIGHT", self.max_height, MAX_CONFIGURED_HEIGHT),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} debe ser un entero")
            if not 2 <= value <= ceiling:
                raise ValueError(f"{name} debe estar entre 2 y {ceiling}")

    @classmethod
    def from_environment(
        cls, max_width: int | None = None, max_height: int | None = None,
    ) -> "MpcVideoProfile":
        """Argumentos sobre entorno; el alto predeterminado es 3/4 del ancho.

        Define una caja máxima, no la proporción del video. El preprocesador
        conserva la proporción y no amplía. Los servicios usan el mismo perfil.
        Aceptar el límite no demuestra rendimiento de 20 s a 640x480.
        """
        width = cls._dimension("MPC_MAX_WIDTH", max_width, DEFAULT_MPC_WIDTH)
        height = cls._dimension("MPC_MAX_HEIGHT", max_height, width * 3 // 4)
        return cls(width, height)

    @staticmethod
    def _dimension(name: str, override: int | None, default: int) -> int:
        if override is not None:
            return override
        value = os.getenv(name)
        if value is None or not value.strip():
            return default
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(f"{name} debe ser un entero") from exc

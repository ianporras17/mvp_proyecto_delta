"""Límites de video compartidos por el cliente y las APIs de control."""

from dataclasses import dataclass
import os


# Cotas de entrada, no una promesa de tiempo real o memoria ilimitada.
DEFAULT_MPC_WIDTH = 1920
MAX_CONFIGURED_WIDTH = 1920
MAX_CONFIGURED_HEIGHT = 1080
DEFAULT_MPC_FPS = 30.0
MAX_CONFIGURED_FPS = 30


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
        cls,
        max_width: int | None = None,
        max_height: int | None = None,
    ) -> "MpcVideoProfile":
        """Los argumentos tienen prioridad; la caja nueva por defecto es 1080p.

        Este par define una caja máxima, no la proporción del video. Todos los
        servicios deben configurar los mismos máximos antes de aceptar un job.
        Los perfiles menores conservan el alto 4:3 anterior hasta 1080.
        Admitir una resolución no demuestra su rendimiento en videos largos.
        """
        width = cls._dimension("MPC_MAX_WIDTH", max_width, DEFAULT_MPC_WIDTH)
        height = cls._dimension("MPC_MAX_HEIGHT", max_height, min(1080, width * 3 // 4))
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

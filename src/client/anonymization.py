"""Public region plan for MPC black rectangles; original pixels stay unchanged.

The client builds only small public 0/1 mask chunks. MP-SPDZ applies those
masks to secret pixels; this module does not create an anonymized video.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cached_property
from numbers import Integral
from types import MappingProxyType

from client.regions import FrameDetections, SensitiveRegion, prepare_regions


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} debe ser un entero mayor o igual a {minimum}")
    return int(value)


@dataclass(frozen=True)
class AnonymizationPlan:
    """Validated half-open rectangles in processed grayscale frame coordinates."""

    width: int
    height: int
    frame_count: int
    regions: tuple[SensitiveRegion, ...]
    _detections_complete: bool = field(default=False, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        for name in ("width", "height", "frame_count"):
            object.__setattr__(self, name, _integer(getattr(self, name), name, minimum=1))
        try:
            regions = tuple(self.regions)
        except TypeError as exc:
            raise ValueError("regions debe contener SensitiveRegion") from exc
        for region in regions:
            if not isinstance(region, SensitiveRegion):
                raise ValueError("regions solo admite SensitiveRegion")
            if region.frame_index >= self.frame_count:
                raise ValueError("frame_index fuera de los cuadros procesados")
            if region.x + region.width > self.width or region.y + region.height > self.height:
                raise ValueError("La region excede las dimensiones del cuadro procesado")
        object.__setattr__(self, "regions", regions)

    @property
    def payload_size(self) -> int:
        return self.width * self.height * self.frame_count

    @property
    def detections_complete(self) -> bool:
        """La fábrica verificó un resultado explícito por cada cuadro procesado."""
        return self._detections_complete

    @cached_property
    def _regions_by_frame(self) -> Mapping[int, tuple[SensitiveRegion, ...]]:
        # Index only rectangles, never allocate a width*height*frame_count mask.
        grouped: dict[int, list[SensitiveRegion]] = {}
        for region in self.regions:
            grouped.setdefault(region.frame_index, []).append(region)
        return MappingProxyType({index: tuple(regions) for index, regions in grouped.items()})

    def mask_chunk(self, offset: int, length: int) -> bytes:
        """Return 0=keep/1=hide in frame-major, row-major pixel order.

        Chunks can cross rows and frames. Mixed/overlapping rectangles use the
        same union, independent of their face/plate label.
        """
        offset = _integer(offset, "offset")
        length = _integer(length, "length")
        end = offset + length
        if offset > self.payload_size or end > self.payload_size:
            raise ValueError("El bloque de mascara excede los pixeles del video")
        if not length or not self.regions:
            return bytes(length)

        mask = bytearray(length)
        frame_size = self.width * self.height
        first_frame = offset // frame_size
        last_frame = (end - 1) // frame_size
        for frame_index in range(first_frame, last_frame + 1):
            frame_offset = frame_index * frame_size
            local_begin = max(0, offset - frame_offset)
            local_end = min(frame_size, end - frame_offset)
            first_row = local_begin // self.width
            last_row = (local_end - 1) // self.width
            for region in self._regions_by_frame.get(frame_index, ()):
                row_start = max(first_row, region.y)
                row_end = min(last_row + 1, region.y + region.height)
                for row in range(row_start, row_end):
                    rectangle_start = frame_offset + row * self.width + region.x
                    begin = max(offset, rectangle_start)
                    stop = min(end, rectangle_start + region.width)
                    if stop > begin:
                        mask[begin - offset:stop - offset] = b"\x01" * (stop - begin)
        return bytes(mask)

    def validate_result(self, original: bytes, result: bytes, *, chunk_size: int = 4096) -> None:
        """Verify unchanged outside pixels and zero inside; never modify inputs."""
        chunk_size = _integer(chunk_size, "chunk_size", minimum=1)
        if not isinstance(original, bytes) or not isinstance(result, bytes):
            raise ValueError("original y result deben ser bytes")
        if len(original) != self.payload_size or len(result) != self.payload_size:
            raise ValueError("La cantidad de pixeles no coincide con el plan")
        for offset in range(0, self.payload_size, chunk_size):
            mask = self.mask_chunk(offset, min(chunk_size, self.payload_size - offset))
            for position, hidden in enumerate(mask, start=offset):
                expected = 0 if hidden else original[position]
                if result[position] != expected:
                    raise ValueError(f"Resultado MPC incorrecto en el pixel {position}")


def prepare_anonymization(
    frames: Iterable[FrameDetections],
    *,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    frame_count: int,
) -> AnonymizationPlan:
    """Prepare every detection result before any pixels can be transmitted."""
    frame_count = _integer(frame_count, "frame_count", minimum=1)
    regions = prepare_regions(
        frames, source_size=source_size, target_size=target_size, frame_count=frame_count,
    )
    width, height = target_size
    plan = AnonymizationPlan(width, height, frame_count, regions)
    object.__setattr__(plan, "_detections_complete", True)
    return plan

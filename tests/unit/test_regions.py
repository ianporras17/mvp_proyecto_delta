"""Contrato T024: regiones públicas por cuadro y geometría conservadora."""

from dataclasses import FrozenInstanceError, asdict, fields

import numpy as np
import pytest

from client.regions import (
    DetectedRegion,
    DetectorExecutionError,
    FrameDetections,
    RegionType,
    SensitiveRegion,
    logical_batch_index,
    prepare_frame_regions,
    prepare_regions,
    transform_region,
)


def _face(x=400, y=200, width=100, height=60) -> DetectedRegion:
    return DetectedRegion(RegionType.FACE, x, y, width, height)


def test_final_contract_is_typed_immutable_and_has_only_the_six_public_fields() -> None:
    region = SensitiveRegion(0, RegionType.FACE, 200, 100, 50, 30)

    assert [field.name for field in fields(region)] == [
        "frame_index", "region_type", "x", "y", "width", "height"
    ]
    assert asdict(region) == {
        "frame_index": 0, "region_type": "face", "x": 200, "y": 100,
        "width": 50, "height": 30,
    }
    assert all(type(getattr(region, name)) is int for name in (
        "frame_index", "x", "y", "width", "height"
    ))
    assert RegionType.FACE.value == "face"
    assert RegionType.PLATE.value == "plate"
    with pytest.raises(FrozenInstanceError):
        region.x = 201


@pytest.mark.parametrize(
    ("source_size", "target_size", "box", "expected"),
    [
        ((1280, 720), (640, 360), (400, 200, 100, 60), (200, 100, 50, 30)),
        ((1920, 1080), (640, 360), (600, 300, 150, 90), (200, 100, 50, 30)),
        ((368, 368), (368, 368), (100, 80, 45, 60), (100, 80, 45, 60)),
        ((320, 240), (320, 240), (10, 20, 30, 40), (10, 20, 30, 40)),
        ((642, 482), (478, 360), (10.1, 20.2, 30.3, 40.4), (7, 15, 24, 31)),
    ],
)
def test_transforms_using_actual_source_and_destination_dimensions(
    source_size: tuple[int, int], target_size: tuple[int, int],
    box: tuple, expected: tuple[int, int, int, int],
) -> None:
    region = transform_region(
        _face(*box), frame_index=3, source_size=source_size, target_size=target_size
    )

    assert region == SensitiveRegion(3, RegionType.FACE, *expected)


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        ((0, 0, 100, 100), (0, 0, 50, 50)),
        ((98, 98, 2, 2), (49, 49, 1, 1)),
        ((-15.2, -4.4, 35.3, 20.1), (0, 0, 11, 8)),
        ((90.1, 90.1, 20, 30), (45, 45, 5, 5)),
        ((-10, -10, 120, 120), (0, 0, 50, 50)),
        ((10.1, 20.1, 0.1, 0.1), (5, 10, 1, 1)),
    ],
)
def test_clips_borders_and_rounds_outward_with_exclusive_endpoints(
    box: tuple, expected: tuple[int, int, int, int],
) -> None:
    region = transform_region(
        _face(*box), frame_index=0, source_size=(100, 100), target_size=(50, 50)
    )

    assert region == SensitiveRegion(0, RegionType.FACE, *expected)
    assert 0 <= region.x < region.x + region.width <= 50
    assert 0 <= region.y < region.y + region.height <= 50


def test_positive_subpixel_box_is_not_dropped_by_float_endpoint_roundoff() -> None:
    region = transform_region(
        _face(2, 2, 1e-17, 1e-17), frame_index=0,
        source_size=(4, 4), target_size=(2, 2),
    )
    assert region == SensitiveRegion(0, RegionType.FACE, 1, 1, 1, 1)


def test_exact_integer_boundary_does_not_gain_an_unnecessary_extra_pixel() -> None:
    region = transform_region(
        _face(3, 3, 3, 3), frame_index=0,
        source_size=(6, 6), target_size=(2, 2),
    )
    assert region == SensitiveRegion(0, RegionType.FACE, 1, 1, 1, 1)


@pytest.mark.parametrize("box", [
    (100, 10, 10, 10), (-10, 10, 10, 10),
    (10, 100, 10, 10), (10, -10, 10, 10),
    (150, 150, 20, 20), (-50, -50, 10, 10),
])
def test_valid_boxes_wholly_outside_or_only_touching_are_omitted(box: tuple) -> None:
    assert transform_region(
        _face(*box), frame_index=0, source_size=(100, 100), target_size=(50, 50)
    ) is None


def test_one_frame_can_contain_multiple_faces_and_plates_in_detector_order() -> None:
    frame = FrameDetections.success(2, (
        _face(),
        _face(600, 300, 80, 40),
        DetectedRegion(RegionType.PLATE, 800, 400, 100, 60),
    ))

    result = prepare_frame_regions(
        frame, source_size=(1280, 720), target_size=(640, 360)
    )

    assert result == (
        SensitiveRegion(2, RegionType.FACE, 200, 100, 50, 30),
        SensitiveRegion(2, RegionType.FACE, 300, 150, 40, 20),
        SensitiveRegion(2, RegionType.PLATE, 400, 200, 50, 30),
    )
    assert isinstance(result, tuple)


def test_successful_empty_frame_is_not_a_detector_failure() -> None:
    empty = FrameDetections.success(0)
    failed = FrameDetections.failure(0, "YuNet no pudo ejecutarse")

    assert empty.succeeded is True
    assert empty.error is None
    assert empty.detections == ()
    assert prepare_frame_regions(
        empty, source_size=(1280, 720), target_size=(640, 360)
    ) == ()
    assert failed.succeeded is False
    assert failed.detections == ()
    with pytest.raises(DetectorExecutionError, match="YuNet"):
        prepare_frame_regions(
            failed, source_size=(1280, 720), target_size=(640, 360)
        )


def test_processed_frame_indices_are_retained_even_when_other_frames_are_empty() -> None:
    frames = (
        FrameDetections.success(0),
        FrameDetections.success(1, (_face(),)),
        FrameDetections.success(2),
        FrameDetections.success(3, (
            DetectedRegion(RegionType.PLATE, 800, 400, 100, 60),
        )),
    )

    result = prepare_regions(
        frames, source_size=(1280, 720), target_size=(640, 360)
    )

    assert isinstance(result, tuple)
    assert result == (
        SensitiveRegion(1, RegionType.FACE, 200, 100, 50, 30),
        SensitiveRegion(3, RegionType.PLATE, 400, 200, 50, 30),
    )


@pytest.mark.parametrize("indices", [(1,), (0, 2), (0, 0), (0, 2, 1), (0, 3, 6)])
def test_requires_contiguous_processed_indices_not_source_or_batch_ids(
    indices: tuple[int, ...],
) -> None:
    frames = tuple(FrameDetections.success(index) for index in indices)

    with pytest.raises(ValueError):
        prepare_regions(frames, source_size=(1280, 720), target_size=(640, 360))


def test_later_detector_failure_prevents_returning_a_partial_result() -> None:
    frames = (
        FrameDetections.success(0, (_face(),)),
        FrameDetections.success(1),
        FrameDetections.failure(2, "falló el detector de placas"),
    )

    with pytest.raises(DetectorExecutionError, match="detector de placas"):
        prepare_regions(frames, source_size=(1280, 720), target_size=(640, 360))


def test_empty_video_sequence_returns_no_regions_but_still_checks_dimensions() -> None:
    assert prepare_regions((), source_size=(368, 368), target_size=(368, 368)) == ()
    with pytest.raises(ValueError):
        prepare_regions((), source_size=(0, 368), target_size=(368, 368))


def test_frame_count_checks_presence_of_all_processed_frames_including_empty_ones() -> None:
    frames = (FrameDetections.success(0), FrameDetections.success(1))
    assert prepare_regions(
        frames, source_size=(368, 368), target_size=(368, 368), frame_count=2
    ) == ()

    for frame_count in (1, 3):
        with pytest.raises(ValueError):
            prepare_regions(
                frames, source_size=(368, 368), target_size=(368, 368),
                frame_count=frame_count,
            )
    with pytest.raises(ValueError):
        prepare_regions(
            (), source_size=(368, 368), target_size=(368, 368), frame_count=1
        )


@pytest.mark.parametrize("frame_count", [0, -1, True, False, 1.5, "1"])
def test_declared_frame_count_must_be_a_positive_integer(frame_count) -> None:
    with pytest.raises(ValueError):
        prepare_regions(
            (FrameDetections.success(0),), source_size=(368, 368),
            target_size=(368, 368), frame_count=frame_count,
        )


def test_numpy_integral_values_are_normalized_to_python_integers() -> None:
    region = SensitiveRegion(
        np.int64(1), RegionType.FACE, np.int64(2), np.int64(3),
        np.int64(4), np.int64(5),
    )
    assert all(type(getattr(region, name)) is int for name in (
        "frame_index", "x", "y", "width", "height"
    ))
    transformed = transform_region(
        _face(), frame_index=np.int64(1),
        source_size=(np.int64(1280), np.int64(720)),
        target_size=(np.int64(640), np.int64(360)),
    )
    assert transformed == SensitiveRegion(1, RegionType.FACE, 200, 100, 50, 30)


@pytest.mark.parametrize("invalid_size", [
    (0, 720), (1280, 0), (-1, 720), (1280, -1),
    (True, 720), (1280, False), (1280.5, 720),
    (float("nan"), 720), (1280, float("inf")),
    (1280,), (1280, 720, 1),
])
@pytest.mark.parametrize("side", ["source", "target"])
def test_invalid_dimensions_are_rejected_even_without_detections(
    invalid_size: tuple, side: str,
) -> None:
    source_size = invalid_size if side == "source" else (1280, 720)
    target_size = invalid_size if side == "target" else (640, 360)

    with pytest.raises(ValueError):
        transform_region(
            _face(), frame_index=0, source_size=source_size, target_size=target_size
        )
    with pytest.raises(ValueError):
        prepare_frame_regions(
            FrameDetections.success(0), source_size=source_size, target_size=target_size
        )


@pytest.mark.parametrize("box", [
    (0, 0, 0, 10), (0, 0, 10, 0), (0, 0, -1, 10), (0, 0, 10, -1),
    (float("nan"), 0, 10, 10), (0, float("nan"), 10, 10),
    (0, 0, float("nan"), 10), (0, 0, 10, float("nan")),
    (float("inf"), 0, 10, 10), (0, float("-inf"), 10, 10),
    (0, 0, float("inf"), 10), (0, 0, 10, float("inf")),
    (1e308, 0, 1e308, 10), (0, 1e308, 10, 1e308),
])
def test_invalid_or_overflowing_detection_geometry_is_not_silently_dropped(
    box: tuple,
) -> None:
    with pytest.raises(ValueError):
        transform_region(
            _face(*box), frame_index=0, source_size=(1280, 720), target_size=(640, 360)
        )


@pytest.mark.parametrize("invalid_index", [-1, True, False, 0.5, "0"])
def test_indices_are_nonnegative_integers_and_never_booleans(invalid_index) -> None:
    with pytest.raises(ValueError):
        SensitiveRegion(invalid_index, RegionType.FACE, 0, 0, 10, 10)
    with pytest.raises(ValueError):
        FrameDetections.success(invalid_index)
    with pytest.raises(ValueError):
        transform_region(
            _face(), frame_index=invalid_index,
            source_size=(1280, 720), target_size=(640, 360),
        )
    with pytest.raises(ValueError):
        logical_batch_index(invalid_index)


@pytest.mark.parametrize(("field", "invalid_value"), [
    ("x", -1), ("y", -1), ("x", True), ("y", 0.5),
    ("width", 0), ("height", 0), ("width", -1), ("height", -1),
    ("width", True), ("height", 1.5),
])
def test_final_coordinates_are_strict_integers_with_positive_area(
    field: str, invalid_value,
) -> None:
    values = dict(frame_index=0, region_type=RegionType.FACE, x=0, y=0, width=10, height=10)
    values[field] = invalid_value
    with pytest.raises(ValueError):
        SensitiveRegion(**values)


@pytest.mark.parametrize("region_type", ["car", "", None, 0])
def test_unrecognized_region_types_are_rejected(region_type) -> None:
    with pytest.raises(ValueError):
        DetectedRegion(region_type, 0, 0, 10, 10)
    with pytest.raises(ValueError):
        SensitiveRegion(0, region_type, 0, 0, 10, 10)


@pytest.mark.parametrize("error", ["", " ", "\n\t", None])
def test_detector_failure_requires_a_nonempty_error(error) -> None:
    with pytest.raises(ValueError):
        FrameDetections.failure(0, error)


def test_failure_cannot_also_claim_successful_regions() -> None:
    with pytest.raises(ValueError):
        FrameDetections(0, (_face(),), "falló el detector")


@pytest.mark.parametrize("detections", [None, (object(),)])
def test_success_rejects_missing_or_untyped_detection_collections(detections) -> None:
    with pytest.raises(ValueError):
        FrameDetections.success(0, detections)


@pytest.mark.parametrize(("frame_index", "expected"), [
    (0, 0), (9, 0), (10, 1), (19, 1), (20, 2), (29, 2), (30, 3),
])
def test_logical_batches_of_ten_never_replace_the_individual_frame_index(
    frame_index: int, expected: int,
) -> None:
    region = SensitiveRegion(frame_index, RegionType.FACE, 0, 0, 10, 10)
    assert logical_batch_index(region.frame_index) == expected
    assert region.frame_index == frame_index
    assert "batch_id" not in asdict(region)


def test_logical_batch_size_can_be_changed_without_transport_changes() -> None:
    assert logical_batch_index(9, batch_size=5) == 1
    assert logical_batch_index(10, batch_size=5) == 2


@pytest.mark.parametrize("batch_size", [0, -1, True, False, 1.5, "10"])
def test_logical_batch_size_must_be_a_positive_integer(batch_size) -> None:
    with pytest.raises(ValueError):
        logical_batch_index(0, batch_size=batch_size)

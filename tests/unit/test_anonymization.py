"""T025 public chunk masks and validation; no client-side pixel anonymization."""

from dataclasses import FrozenInstanceError, fields

import numpy as np
import pytest

from client.anonymization import AnonymizationPlan, prepare_anonymization
from client.regions import (
    DetectedRegion,
    DetectorExecutionError,
    FrameDetections,
    RegionType,
    SensitiveRegion,
)


def _region(frame_index=0, x=1, y=1, width=2, height=2, kind=RegionType.FACE):
    return SensitiveRegion(frame_index, kind, x, y, width, height)


def _expected_mask(plan: AnonymizationPlan) -> bytes:
    """Small independent test oracle, not a production full-video mask."""
    pixels = np.zeros((plan.frame_count, plan.height, plan.width), dtype=np.uint8)
    for region in plan.regions:
        pixels[
            region.frame_index,
            region.y:region.y + region.height,
            region.x:region.x + region.width,
        ] = 1
    return pixels.tobytes()


def test_plan_is_frozen_has_four_fields_and_copies_mutable_region_collection() -> None:
    regions = [_region()]
    plan = AnonymizationPlan(4, 3, 2, regions)
    regions.clear()

    assert [field.name for field in fields(plan) if field.init] == ["width", "height", "frame_count", "regions"]
    assert plan.regions == (_region(),)
    assert isinstance(plan.regions, tuple)
    assert plan.payload_size == 24
    with pytest.raises(FrozenInstanceError):
        plan.width = 8


@pytest.mark.parametrize("field", ["width", "height", "frame_count"])
@pytest.mark.parametrize("value", [0, -1, True, False, 1.0, "4", None])
def test_plan_rejects_invalid_dimensions_and_counts(field: str, value) -> None:
    arguments = {"width": 4, "height": 3, "frame_count": 2, "regions": ()}
    arguments[field] = value
    with pytest.raises(ValueError):
        AnonymizationPlan(**arguments)


def test_integer_scalars_are_normalized_to_python_integers() -> None:
    plan = AnonymizationPlan(np.int64(4), np.int32(3), np.int64(2), ())
    assert type(plan.width) is type(plan.height) is type(plan.frame_count) is int
    assert plan.mask_chunk(np.int64(20), np.int32(4)) == b"\x00" * 4


@pytest.mark.parametrize("regions", [None, 5, (_region(), None), ({"x": 1},), (DetectedRegion("face", 1, 1, 2, 2),)])
def test_plan_requires_sensitive_regions(regions) -> None:
    with pytest.raises(ValueError):
        AnonymizationPlan(4, 3, 2, regions)


@pytest.mark.parametrize(
    "region",
    [_region(frame_index=2), _region(x=3, width=2), _region(y=2, height=2)],
)
def test_plan_rejects_out_of_bounds_frame_or_rectangle(region: SensitiveRegion) -> None:
    with pytest.raises(ValueError):
        AnonymizationPlan(4, 3, 2, (region,))


def test_no_regions_preserves_all_pixels_and_all_frames() -> None:
    plan = prepare_anonymization(
        (FrameDetections.success(0), FrameDetections.success(1)),
        source_size=(8, 6), target_size=(4, 3), frame_count=2,
    )
    original = bytes(range(24))
    assert plan.regions == ()
    assert plan.mask_chunk(0, 24) == bytes(24)
    plan.validate_result(original, original)


@pytest.mark.parametrize("kind", [RegionType.FACE, RegionType.PLATE])
def test_region_types_share_identical_generic_half_open_mask(kind: RegionType) -> None:
    plan = AnonymizationPlan(4, 3, 1, (_region(kind=kind),))
    assert plan.mask_chunk(0, 12) == bytes([
        0, 0, 0, 0,
        0, 1, 1, 0,
        0, 1, 1, 0,
    ])


def test_mixed_and_overlapping_regions_use_union_never_double_count_or_toggle() -> None:
    plan = AnonymizationPlan(5, 4, 1, (
        _region(x=0, y=0, width=3, height=3),
        _region(x=2, y=1, width=3, height=2, kind=RegionType.PLATE),
        _region(x=0, y=0, width=3, height=3),
    ))
    assert plan.mask_chunk(0, 20) == bytes([
        1, 1, 1, 0, 0,
        1, 1, 1, 1, 1,
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0,
    ])


def test_rectangle_touching_every_image_border_covers_exactly_the_image() -> None:
    plan = AnonymizationPlan(4, 3, 2, (_region(1, 0, 0, 4, 3),))
    assert plan.mask_chunk(0, 24) == bytes(12) + b"\x01" * 12


def test_single_pixel_at_bottom_right_does_not_extend_to_next_frame() -> None:
    plan = AnonymizationPlan(4, 3, 2, (_region(0, 3, 2, 1, 1),))
    assert plan.mask_chunk(10, 4) == bytes([0, 1, 0, 0])


@pytest.mark.parametrize(
    ("source_size", "target_size", "box", "expected"),
    [
        ((1280, 720), (640, 360), (400, 200, 100, 60), (200, 100, 50, 30)),
        ((1920, 1080), (640, 360), (600, 300, 150, 90), (200, 100, 50, 30)),
        ((368, 368), (368, 368), (100, 80, 45, 60), (100, 80, 45, 60)),
        ((642, 482), (478, 360), (10.1, 20.2, 30.3, 40.4), (7, 15, 24, 31)),
        ((100, 100), (50, 50), (-15.2, -4.4, 35.3, 20.1), (0, 0, 11, 8)),
    ],
)
def test_preparation_uses_t024_actual_dimensions_and_conservative_clipping(
    source_size, target_size, box, expected,
) -> None:
    plan = prepare_anonymization(
        (FrameDetections.success(0, (DetectedRegion(RegionType.FACE, *box),)),),
        source_size=source_size, target_size=target_size, frame_count=1,
    )
    assert plan.regions == (_region(0, *expected),)
    assert (plan.width, plan.height) == target_size


def test_tiny_subpixel_detection_stays_covered_after_resize() -> None:
    plan = prepare_anonymization(
        (FrameDetections.success(0, (DetectedRegion("face", 2, 2, 1e-17, 1e-17),)),),
        source_size=(4, 4), target_size=(2, 2), frame_count=1,
    )
    assert plan.mask_chunk(0, 4) == bytes([0, 0, 0, 1])


def test_fully_exterior_valid_rectangle_is_omitted_without_detector_failure() -> None:
    plan = prepare_anonymization(
        (FrameDetections.success(0, (DetectedRegion("plate", 100, 100, 10, 10),)),),
        source_size=(10, 10), target_size=(4, 4), frame_count=1,
    )
    assert plan.regions == ()
    assert plan.mask_chunk(0, 16) == bytes(16)


def test_masks_cross_rows_frames_and_4096_pixel_chunk_boundaries() -> None:
    plan = AnonymizationPlan(101, 83, 3, (
        _region(0, 90, 40, 11, 43),
        _region(1, 0, 0, 20, 5, RegionType.PLATE),
        _region(1, 4, 2, 93, 70),
        _region(2, 98, 81, 3, 2, RegionType.PLATE),
    ))
    expected = _expected_mask(plan)
    chunks = [plan.mask_chunk(offset, min(4096, plan.payload_size - offset))
              for offset in range(0, plan.payload_size, 4096)]
    assert b"".join(chunks) == expected
    assert set(expected) == {0, 1}
    for offset, length in [(4030, 90), (8380, 14), (16764, 10), (0, 1), (plan.payload_size - 1, 1)]:
        assert plan.mask_chunk(offset, length) == expected[offset:offset + length]


def test_many_irregular_chunks_equal_one_full_mask_on_small_fixture() -> None:
    plan = AnonymizationPlan(7, 5, 4, (
        _region(0, 1, 1, 4, 3), _region(2, 0, 0, 7, 1, RegionType.PLATE),
        _region(3, 6, 4, 1, 1),
    ))
    expected = _expected_mask(plan)
    for size in (1, 3, 7, 11, 35, 39, 1000):
        actual = b"".join(plan.mask_chunk(offset, min(size, plan.payload_size - offset))
                          for offset in range(0, plan.payload_size, size))
        assert actual == expected


def test_large_video_plan_allocates_only_requested_mask_and_sparse_rectangle_index() -> None:
    plan = AnonymizationPlan(640, 480, 200, (_region(199, 639, 479, 1, 1),))
    chunk = plan.mask_chunk(plan.payload_size - 4096, 4096)
    assert len(chunk) == 4096
    assert sum(chunk) == 1
    assert chunk[-1] == 1
    assert set(plan._regions_by_frame) == {199}
    with pytest.raises(TypeError):
        plan._regions_by_frame[0] = ()
    assert not any(isinstance(value, (bytes, bytearray)) for value in vars(plan).values())


@pytest.mark.parametrize(
    ("offset", "length"),
    [(-1, 1), (0, -1), (True, 1), (0, False), (0.0, 1), (0, 1.0),
     ("0", 1), (0, None), (12, 1), (13, 0), (11, 2)],
)
def test_invalid_mask_ranges_are_rejected(offset, length) -> None:
    with pytest.raises(ValueError):
        AnonymizationPlan(4, 3, 1, (_region(),)).mask_chunk(offset, length)


def test_zero_length_mask_is_valid_at_start_middle_and_exclusive_end() -> None:
    plan = AnonymizationPlan(4, 3, 1, (_region(),))
    assert plan.mask_chunk(0, 0) == plan.mask_chunk(7, 0) == plan.mask_chunk(12, 0) == b""


def test_preparation_requires_a_success_result_for_every_frame_even_when_empty() -> None:
    results = (FrameDetections.success(index) for index in range(3))
    plan = prepare_anonymization(
        results, source_size=(4, 3), target_size=(4, 3), frame_count=3,
    )
    assert plan.frame_count == 3
    assert plan.regions == ()


def test_only_full_detection_preparation_proves_execution_for_every_frame() -> None:
    manual = AnonymizationPlan(4, 3, 2, ())
    assert manual.detections_complete is False
    with pytest.raises(ValueError):
        prepare_anonymization(
            (FrameDetections.success(0),),
            source_size=(4, 3), target_size=(4, 3), frame_count=2,
        )
    prepared = prepare_anonymization(
        (FrameDetections.success(0), FrameDetections.success(1)),
        source_size=(4, 3), target_size=(4, 3), frame_count=2,
    )
    assert prepared.detections_complete is True
    assert prepared.regions == manual.regions == ()


@pytest.mark.parametrize(
    "frames",
    [(), (FrameDetections.success(0),),
     (FrameDetections.success(0), FrameDetections.success(2)),
     (FrameDetections.success(0), FrameDetections.success(0)),
     (FrameDetections.success(1), FrameDetections.success(0)),
     (FrameDetections.success(0), FrameDetections.success(1), FrameDetections.success(2))],
)
def test_missing_extra_duplicate_or_out_of_order_detection_results_are_rejected(frames) -> None:
    with pytest.raises(ValueError):
        prepare_anonymization(frames, source_size=(4, 3), target_size=(4, 3), frame_count=2)


def test_failed_detector_aborts_entire_preparation_instead_of_returning_partial_regions() -> None:
    frames = (
        FrameDetections.success(0, (DetectedRegion("face", 0, 0, 2, 2),)),
        FrameDetections.failure(1, "YuNet failed"),
    )
    with pytest.raises(DetectorExecutionError, match="frame_index=1.*YuNet failed"):
        prepare_anonymization(frames, source_size=(4, 3), target_size=(4, 3), frame_count=2)


@pytest.mark.parametrize("frame_count", [None, 0, -1, True, 2.0])
def test_preparation_rejects_invalid_required_count_before_consuming_detector_results(frame_count) -> None:
    consumed = False

    def detections():
        nonlocal consumed
        consumed = True
        yield FrameDetections.success(0)

    with pytest.raises(ValueError):
        prepare_anonymization(detections(), source_size=(4, 3), target_size=(4, 3), frame_count=frame_count)
    assert consumed is False


def test_result_validation_checks_mpc_output_without_modifying_or_returning_original_pixels() -> None:
    plan = AnonymizationPlan(4, 3, 2, (_region(), _region(1, 0, 0, 1, 1, RegionType.PLATE)))
    original = bytes(range(1, 25))
    snapshot = original[:]
    # This masked fixture is only the expected server result, not a client path.
    result = bytes(0 if flag else pixel for pixel, flag in zip(original, _expected_mask(plan)))
    assert plan.validate_result(original, result, chunk_size=5) is None
    assert original == snapshot
    assert original != result


@pytest.mark.parametrize("position", [0, 5, 6, 9, 10, 11])
def test_result_validation_rejects_modified_outside_or_nonzero_inside_pixels(position: int) -> None:
    plan = AnonymizationPlan(4, 3, 1, (_region(),))
    original = bytes(range(1, 13))
    result = bytearray(0 if flag else pixel for pixel, flag in zip(original, _expected_mask(plan)))
    result[position] = 200
    with pytest.raises(ValueError, match=f"pixel {position}"):
        plan.validate_result(original, bytes(result), chunk_size=5)


@pytest.mark.parametrize(("original", "result"), [(b"", bytes(12)), (bytes(12), b""), (bytes(13), bytes(13)), (bytearray(12), bytes(12)), (bytes(12), bytearray(12))])
def test_result_validation_rejects_incorrect_length_or_nonbytes(original, result) -> None:
    with pytest.raises(ValueError):
        AnonymizationPlan(4, 3, 1, ()).validate_result(original, result)


@pytest.mark.parametrize("chunk_size", [0, -1, False, 1.5, None])
def test_result_validation_rejects_invalid_chunk_size(chunk_size) -> None:
    with pytest.raises(ValueError):
        AnonymizationPlan(4, 3, 1, ()).validate_result(bytes(12), bytes(12), chunk_size=chunk_size)

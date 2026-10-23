"""Decoder y contrato de placas, con smoke ONNX real y entradas BGR nativas."""

import cv2
import numpy as np
import pytest

from client.detection import LPDYuNet, PlateDetection, PlateDetectionError, to_frame_detections
from client.detection.lpd_yunet import DEFAULT_MODEL_PATH, MODEL_SHA256
from client.regions import DetectorExecutionError, RegionType, prepare_regions


def _outputs(size, cases=()):
    width, height = size
    priors = LPDYuNet._generate_priors(width, height)
    loc = np.zeros((len(priors), 14), dtype=np.float32)
    conf = np.tile((1.0, 0.0), (len(priors), 1)).astype(np.float32)
    iou = np.ones((len(priors), 1), dtype=np.float32)
    for index, points, confidence in cases:
        center = priors[index, :2] * (width, height)
        anchor = priors[index, 2:] * (width, height)
        for start, point in zip((4, 6, 10, 12), points):
            loc[index, start:start + 2] = (np.asarray(point) - center) / (0.1 * anchor)
        conf[index] = (1 - confidence ** 2, confidence ** 2)
    return loc, conf, iou


def _rectangle(left, top, right, bottom):
    return ((left, top), (right, top), (right, bottom), (left, bottom))


class FakeNet:
    def __init__(self, cases=()):
        self.cases = cases
        self.blob = None
        self.output_override = None
        self.error = None

    def setPreferableBackend(self, backend):
        self.backend = backend

    def setPreferableTarget(self, target):
        self.target = target

    def setInput(self, blob):
        self.blob = blob

    def forward(self, names):
        assert names == ["loc", "conf", "iou"]
        if self.error:
            raise self.error
        if self.output_override is not None:
            return self.output_override
        return _outputs((self.blob.shape[3], self.blob.shape[2]), self.cases)


@pytest.fixture
def make_detector(tmp_path, monkeypatch):
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"test-resource")

    def create(cases=(), **kwargs):
        net = FakeNet(cases)
        monkeypatch.setattr(cv2.dnn, "readNet", lambda path: net)
        return LPDYuNet(model_path, **kwargs), net

    return create


@pytest.mark.parametrize("size", [(320, 240), (478, 850), (641, 479)])
def test_input_native_dimensions_and_bgr_values(make_detector, size):
    width, height = size
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :, :] = (12, 85, 240)
    detector, net = make_detector()
    assert detector.detect(image) == ()
    assert net.blob.shape == (1, 3, height, width)
    np.testing.assert_array_equal(net.blob[0], image.transpose(2, 0, 1))
    assert net.backend == cv2.dnn.DNN_BACKEND_OPENCV
    assert net.target == cv2.dnn.DNN_TARGET_CPU


@pytest.mark.parametrize("image", [None, np.zeros((24, 32)), np.zeros((24, 32, 3)),
    np.zeros((24, 32, 4), dtype=np.uint8), np.zeros((0, 32, 3), dtype=np.uint8),
    np.zeros((7, 32, 3), dtype=np.uint8)])
def test_invalid_input_is_technical_error(make_detector, image):
    detector, _ = make_detector()
    with pytest.raises(PlateDetectionError, match="BGR uint8"):
        detector.detect(image)


def test_one_plate_original_coordinates_and_confidence(make_detector):
    points = ((40, 30), (90, 25), (95, 50), (38, 55))
    detector, _ = make_detector(((0, points, 0.9),))
    (detection,) = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert np.asarray(detection.quadrilateral) == pytest.approx(np.asarray(points), abs=1e-4)
    assert detection.confidence == pytest.approx(0.9)
    assert (detection.region.x, detection.region.y, detection.region.width, detection.region.height) == pytest.approx((38, 25, 57, 30), abs=1e-4)
    assert detection.region.region_type is RegionType.PLATE


def test_multiple_plates_and_nms_use_xywh_envelopes(make_detector, monkeypatch):
    cases = ((0, _rectangle(20, 30, 80, 50), 0.95),
             (1, _rectangle(120, 80, 190, 110), 0.9))
    detector, _ = make_detector(cases)
    original_nms = cv2.dnn.NMSBoxes
    calls = []

    def record(boxes, scores, *args, **kwargs):
        calls.append(boxes)
        return original_nms(boxes, scores, *args, **kwargs)

    monkeypatch.setattr(cv2.dnn, "NMSBoxes", record)
    detections = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert len(detections) == 2
    assert np.asarray(calls[0]) == pytest.approx(np.asarray(((20, 30, 60, 20), (120, 80, 70, 30))), abs=1e-4)


def test_overlapping_detections_suppressed(make_detector):
    detector, _ = make_detector(((0, _rectangle(20, 30, 80, 50), 0.95),
                                (1, _rectangle(21, 31, 81, 51), 0.9)))
    assert len(detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))) == 1


def test_partial_region_clipped_but_quadrilateral_preserved(make_detector):
    detector, _ = make_detector(((0, _rectangle(-10, -5, 60, 35), 0.9),))
    (detection,) = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert detection.quadrilateral[0] == pytest.approx((-10, -5), abs=1e-4)
    assert (detection.region.x, detection.region.y, detection.region.width, detection.region.height) == pytest.approx((0, 0, 60, 35), abs=1e-4)


def test_valid_box_entirely_outside_omitted(make_detector):
    detector, _ = make_detector(((0, _rectangle(-100, -50, -60, -30), 0.9),))
    assert detector.detect(np.zeros((240, 320, 3), dtype=np.uint8)) == ()


@pytest.mark.parametrize("threshold,expected", [(0.8, 0), (0.6, 1)])
def test_confidence_threshold_configurable(make_detector, threshold, expected):
    detector, _ = make_detector(((0, _rectangle(10, 20, 50, 40), 0.7),), confidence_threshold=threshold)
    assert len(detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))) == expected


def test_project_default_threshold_is_point_six(make_detector):
    detector, _ = make_detector(((0, _rectangle(10, 20, 50, 40), 0.7),))
    assert detector.confidence_threshold == 0.6
    assert len(detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))) == 1


def test_threshold_uses_sqrt_class_times_clipped_iou(make_detector):
    detector, net = make_detector(confidence_threshold=0.5)
    loc, conf, iou = _outputs((320, 240), ((0, _rectangle(10, 20, 50, 40), 0.9),))
    conf[0, 1] = 0.81
    iou[0, 0] = 0.36
    net.output_override = loc, conf, iou
    (detection,) = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert detection.confidence == pytest.approx(0.54)


def test_t024_frame_index_processed_not_batch_and_geometry(make_detector):
    detector, _ = make_detector(((0, _rectangle(100, 50, 200, 100), 0.9),))
    detections = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    frame = to_frame_detections(13, detections)
    assert frame.frame_index == 13
    assert frame.succeeded
    assert frame.detections[0] is detections[0].region
    assert detector.detect_frame(np.zeros((240, 320, 3), dtype=np.uint8), 0).frame_index == 0
    from client.regions import prepare_frame_regions
    (region,) = prepare_frame_regions(frame, source_size=(320, 240), target_size=(160, 120))
    assert region.frame_index == 13
    # La inferencia FP32 puede devolver 100.000007; T024 redondea hacia afuera
    # correctamente, no hay que redondear las salidas del detector hacia dentro.
    assert region.x <= 50 and region.y <= 25
    assert region.x + region.width >= 100 and region.y + region.height >= 50
    assert region.width <= 51 and region.height <= 26


def test_empty_is_success_but_model_failure_is_explicit(make_detector):
    detector, net = make_detector()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    success = detector.detect_frame(image, 0)
    assert success.succeeded and success.detections == ()
    net.error = cv2.error("inference failed")
    with pytest.raises(PlateDetectionError, match="Falló"):
        detector.detect(image)
    failed = detector.detect_frame(image, 0)
    assert not failed.succeeded and "Falló" in failed.error
    with pytest.raises(DetectorExecutionError):
        prepare_regions((failed,), source_size=(320, 240), target_size=(160, 120), frame_count=1)


@pytest.mark.parametrize("mutation", ["shape", "nan", "conf", "degenerate", "outputs"])
def test_invalid_model_output_does_not_become_empty(make_detector, mutation):
    detector, net = make_detector()
    loc, conf, iou = _outputs((320, 240), ((0, _rectangle(10, 20, 50, 40), 0.9),))
    if mutation == "shape":
        loc = loc[:2]
    elif mutation == "nan":
        loc[0, 0] = np.nan
    elif mutation == "conf":
        conf[0, 1] = -1
    elif mutation == "degenerate":
        loc[0] = 0
    net.output_override = (loc, conf) if mutation == "outputs" else (loc, conf, iou)
    with pytest.raises(PlateDetectionError):
        detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))


def test_load_missing_and_broken_model(tmp_path, monkeypatch):
    with pytest.raises(PlateDetectionError, match="No se encontró"):
        LPDYuNet(tmp_path / "missing.onnx")
    path = tmp_path / "broken.onnx"
    path.write_bytes(b"invalid")
    monkeypatch.setattr(cv2.dnn, "readNet", lambda path: (_ for _ in ()).throw(cv2.error("bad model")))
    with pytest.raises(PlateDetectionError, match="No se pudo cargar"):
        LPDYuNet(path)


def test_default_model_hash_mismatch_rejected(tmp_path, monkeypatch):
    from client.detection import lpd_yunet
    path = tmp_path / "tampered.onnx"
    path.write_bytes(b"invalid")
    monkeypatch.setattr(lpd_yunet, "DEFAULT_MODEL_PATH", path)
    with pytest.raises(PlateDetectionError, match="SHA-256"):
        LPDYuNet()


def test_batched_outputs_supported_and_keep_top_k(make_detector):
    detector, net = make_detector(keep_top_k=1)
    outputs = _outputs((320, 240), ((0, _rectangle(10, 20, 50, 40), 0.9),
                                  (1, _rectangle(100, 120, 150, 140), 0.95)))
    net.output_override = tuple(array[np.newaxis] for array in outputs)
    (detection,) = detector.detect(np.zeros((240, 320, 3), dtype=np.uint8))
    assert detection.confidence == pytest.approx(0.95)


def test_t024_empty_result_and_invalid_index(make_detector):
    detector, _ = make_detector()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    frame = to_frame_detections(0, ())
    assert frame.succeeded and frame.detections == ()
    for index in (-1, True, 1.5):
        with pytest.raises(ValueError, match="frame_index"):
            detector.detect_frame(image, index)
    with pytest.raises(ValueError, match="PlateDetection"):
        to_frame_detections(0, ("not a detection",))


@pytest.mark.parametrize("option,value", [("confidence_threshold", -0.1), ("confidence_threshold", 1.1),
    ("confidence_threshold", np.nan), ("nms_threshold", np.inf), ("top_k", 0),
    ("keep_top_k", True), ("keep_top_k", 1.5)])
def test_invalid_options(option, value):
    with pytest.raises(ValueError):
        LPDYuNet(**{option: value})


def test_default_model_hash_and_real_native_onnx_inference():
    import hashlib
    assert hashlib.sha256(DEFAULT_MODEL_PATH.read_bytes()).hexdigest() == MODEL_SHA256
    detector = LPDYuNet()
    for width, height in ((320, 240), (478, 850), (641, 479)):
        result = detector.detect(np.zeros((height, width, 3), dtype=np.uint8))
        assert isinstance(result, tuple)
        assert all(isinstance(detection, PlateDetection) for detection in result)

from pathlib import Path

import cv2
import pytest

from client import cli
from client.preprocessing.yunet import (
    DEFAULT_YUNET_MODEL,
    YuNet,
    YuNetInitializationError,
)


def test_cli_loads_bundled_yunet_once_and_passes_it_to_preprocessing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "video.mp4"
    source.write_bytes(b"placeholder")
    model = object()
    captured = {}

    class FakeYuNet:
        def __init__(self, **kwargs):
            captured["model_path"] = Path(kwargs["modelPath"])
            captured["options"] = kwargs

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "YuNet", FakeYuNet)
    monkeypatch.setattr(cli, "VideoPreprocessor", lambda **kwargs: object())
    monkeypatch.setattr(cli, "CoordinatorClient", lambda url: object())
    monkeypatch.setattr(cli, "PartyClient", lambda: object())
    monkeypatch.setattr(cli, "MpcExternalClient", lambda: object())
    monkeypatch.setattr(cli, "VideoReconstructor", lambda: object())

    def upload_video(**kwargs):
        captured["received_model"] = kwargs["model"]
        return 0

    monkeypatch.setattr(cli, "upload_video", upload_video)
    assert cli.main(["upload", str(source)]) == 0
    assert captured["model_path"] == DEFAULT_YUNET_MODEL
    assert captured["model_path"].is_file()
    assert captured["received_model"] is not None


def test_yunet_reports_missing_or_invalid_model(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    with pytest.raises(YuNetInitializationError, match="No se encontró"):
        YuNet(tmp_path / "missing.onnx")

    monkeypatch.setattr(
        cv2, "FaceDetectorYN", type("BrokenDetector", (), {
            "create": staticmethod(lambda **kwargs: (_ for _ in ()).throw(cv2.error("invalid ONNX")))
        }),
    )
    with pytest.raises(YuNetInitializationError, match="No se pudo cargar.*invalid ONNX"):
        YuNet(DEFAULT_YUNET_MODEL)

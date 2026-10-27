# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import numpy as np
import cv2 as cv
from pathlib import Path


class YuNetInitializationError(RuntimeError):
    """El modelo YuNet no existe o OpenCV no pudo cargarlo."""


DEFAULT_YUNET_MODEL = Path(__file__).resolve().with_name(
    "face_detection_yunet_2023mar.onnx"
)

class YuNet:
    def __init__(self, modelPath, inputSize=[320, 320], confThreshold=0.6, nmsThreshold=0.3, topK=5000, backendId=0, targetId=0):
        self._modelPath = str(Path(modelPath).expanduser().resolve())
        if not Path(self._modelPath).is_file():
            raise YuNetInitializationError(f"No se encontró el modelo YuNet: {self._modelPath}")
        self._inputSize = tuple(inputSize) # [w, h]
        self._confThreshold = confThreshold
        self._nmsThreshold = nmsThreshold
        self._topK = topK
        self._backendId = backendId
        self._targetId = targetId

        self._model = self._load_model()

    def _load_model(self):
        try:
            return cv.FaceDetectorYN.create(
                model=self._modelPath, config="", input_size=self._inputSize,
                score_threshold=self._confThreshold, nms_threshold=self._nmsThreshold,
                top_k=self._topK, backend_id=self._backendId, target_id=self._targetId)
        except cv.error as exc:
            raise YuNetInitializationError(
                f"No se pudo cargar el modelo YuNet '{self._modelPath}': {exc}"
            ) from exc

    @property
    def name(self):
        return self.__class__.__name__

    def setBackendAndTarget(self, backendId, targetId):
        self._backendId = backendId
        self._targetId = targetId
        self._model = self._load_model()

    def setInputSize(self, input_size):
        self._model.setInputSize(tuple(input_size))

    def infer(self, image):
        # Forward
        faces = self._model.detect(image)
        return np.empty(shape=(0, 5)) if faces[1] is None else faces[1]

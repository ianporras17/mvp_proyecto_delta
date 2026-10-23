"""Detección local independiente del procesamiento MPC."""

from client.detection.lpd_yunet import (
    DEFAULT_PLATE_CONFIDENCE_THRESHOLD,
    LPDYuNet,
    PlateDetection,
    PlateDetectionError,
    to_frame_detections,
)

__all__ = ["DEFAULT_PLATE_CONFIDENCE_THRESHOLD", "LPDYuNet", "PlateDetection", "PlateDetectionError", "to_frame_detections"]

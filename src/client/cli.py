"""CLI confiable de identidad y anonimización de rostros/placas con MP-SPDZ."""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import Sequence

import cv2 as cv

from client.preprocessing.yunet import DEFAULT_YUNET_MODEL, YuNet, YuNetInitializationError
from client.detection import DEFAULT_PLATE_CONFIDENCE_THRESHOLD, LPDYuNet, PlateDetectionError

from client.networking import (
    CoordinatorClient,
    CoordinatorClientError,
    PartyClient,
    PartyClientError,
    VideoRegistration,
)
from client.preprocessing import (
    FACE_CONFIDENCE_THRESHOLD,
    VideoPreprocessingError,
    VideoPreprocessor,
    VideoReconstructionError,
    VideoReconstructor,
)
from client.anonymization import AnonymizationPlan, prepare_anonymization
from client.regions import DetectedRegion, DetectorExecutionError, FrameDetections, RegionType
from mpc import MpcExternalClient, MpcExternalIOError
from mpc.video_profile import DEFAULT_MPC_FPS, MAX_CONFIGURED_FPS

backend_target_pairs = (
    (cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU),
    (cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA),
    (cv.dnn.DNN_BACKEND_CUDA, cv.dnn.DNN_TARGET_CUDA_FP16),
    (cv.dnn.DNN_BACKEND_TIMVX, cv.dnn.DNN_TARGET_NPU),
    (cv.dnn.DNN_BACKEND_CANN, cv.dnn.DNN_TARGET_NPU),
)

class ClientInputError(ValueError):
    """La entrada local no cumple el perfil del MVP."""


class MpcResultMismatchError(RuntimeError):
    """El cálculo identidad no devolvió exactamente los pixels enviados."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m client",
        description="Proyecto Delta - MPC de video identidad y anonimización.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser(
        "upload",
        help="Preprocesa un MP4, ejecuta MPC y reconstruye el resultado.",
    )
    upload.add_argument("video", type=Path, help="Ruta local al archivo .mp4")
    upload.add_argument(
        "--processing-mode", choices=("identity", "anonymize"), default="identity",
        help="identity conserva el MVP; anonymize aplica cajas negras MPC a regiones.",
    )
    upload.add_argument(
        "--regions-json", type=Path,
        help="Regiones de prueba/manuales en coordenadas MPC; no integra detectores.",
    )
    upload.add_argument(
        "--detect-plates", action="store_true",
        help="Detecta placas en los cuadros originales; combinable con --detect-faces, requiere anonymize.",
    )
    upload.add_argument(
        "--detect-faces", action="store_true",
        help="Detecta rostros con YuNet; combinable con --detect-plates, requiere anonymize.",
    )
    upload.add_argument(
        "--plate-confidence-threshold", type=float, default=DEFAULT_PLATE_CONFIDENCE_THRESHOLD,
        help="Umbral LPD-YuNet para placas (predeterminado: 0.6); no cambia YuNet.",
    )
    upload.add_argument(
        "--mpc-max-width", type=int, default=None,
        help="Ancho máximo para MPC; predeterminado: MPC_MAX_WIDTH/perfil compartido.",
    )
    upload.add_argument(
        "--mpc-max-height", type=int, default=None,
        help="Alto máximo para MPC; predeterminado: MPC_MAX_HEIGHT o 3/4 del ancho.",
    )
    upload.add_argument(
        "--mpc-fps", type=float, default=DEFAULT_MPC_FPS,
        help=(
            f"FPS de muestreo para MPC; predeterminado: {DEFAULT_MPC_FPS:g}, "
            f"máximo: {MAX_CONFIGURED_FPS}. No aumenta los FPS del original."
        ),
    )
    upload.add_argument(
        "--conf-threshold", "--conf_threshold", type=float,
        default=FACE_CONFIDENCE_THRESHOLD,
        help=f"Confianza mínima de YuNet (predeterminado: {FACE_CONFIDENCE_THRESHOLD:g}).",
    )
    upload.add_argument("--nms-threshold", "--nms_threshold", type=float, default=0.3)
    upload.add_argument("--top-k", "--top_k", type=int, default=5000)
    upload.add_argument(
        "--backend-target",
        "--backend_target",
        "-bt",
        type=int,
        choices=range(len(backend_target_pairs)),
        default=0,
        help=(
            "Backend/target YuNet: 0 OpenCV+CPU (predeterminado), "
            "1 CUDA, 2 CUDA FP16, 3 TIM-VX+NPU, 4 CANN+NPU."
        ),
    )
    return parser


def _validated_video_path(path: Path) -> Path:
    if not path.exists():
        raise ClientInputError(f"El archivo no existe: {path}")
    if not path.is_file():
        raise ClientInputError(f"La ruta no corresponde a un archivo: {path}")
    if path.suffix.lower() != ".mp4":
        raise ClientInputError("El MVP solamente admite archivos .mp4")
    return path


def _plan_from_json(path: Path, metadata) -> AnonymizationPlan:
    """Entrada de diagnóstico; el transporte interno siempre usa tipos T024."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ClientInputError(f"No se pudo leer el JSON de regiones: {exc}") from exc
    if not isinstance(data, dict) or set(data) != {"width", "height", "frames"}:
        raise ClientInputError("El JSON requiere width, height y frames explícitos")
    size = (data["width"], data["height"])
    if size != (metadata.width, metadata.height):
        raise ClientInputError("El JSON debe referirse a las dimensiones MPC reales")
    if not isinstance(data["frames"], list):
        raise ClientInputError("frames debe ser una lista completa de resultados")
    frames = []
    for record in data["frames"]:
        if not isinstance(record, dict):
            raise ClientInputError("Cada fotograma debe declarar su resultado")
        if record.get("status") == "success":
            if set(record) != {"frame_index", "status", "regions"} or not isinstance(record["regions"], list):
                raise ClientInputError("Un éxito requiere frame_index, status y regions, incluso vacías")
            detections = []
            for region in record["regions"]:
                if not isinstance(region, dict) or set(region) != {"region_type", "x", "y", "width", "height"}:
                    raise ClientInputError("Cada región requiere region_type, x, y, width y height")
                detections.append(DetectedRegion(**region))
            frames.append(FrameDetections.success(record["frame_index"], detections))
        elif record.get("status") == "failed":
            if set(record) != {"frame_index", "status", "error"}:
                raise ClientInputError("Un fallo requiere frame_index, status y error")
            frames.append(FrameDetections.failure(record["frame_index"], record["error"]))
        else:
            raise ClientInputError("status debe ser success o failed; no se infiere éxito")
    return prepare_anonymization(
        frames, source_size=size, target_size=size, frame_count=metadata.frame_count
    )


def upload_video(
    video_path: Path,
    coordinator: CoordinatorClient,
    party_client: PartyClient,
    preprocessor: VideoPreprocessor,
    mpc_client: MpcExternalClient,
    reconstructor: VideoReconstructor,
    model: YuNet,
    output_root: str | Path = "output",
    *,
    processing_mode: str = "identity",
    anonymization_plan: AnonymizationPlan | None = None,
    regions_json: Path | None = None,
    detect_plates: bool = False,
    detect_faces: bool = False,
    plate_confidence_threshold: float = DEFAULT_PLATE_CONFIDENCE_THRESHOLD,
) -> int:
    path = _validated_video_path(video_path)
    if processing_mode not in ("identity", "anonymize"):
        raise ClientInputError("processing_mode debe ser identity o anonymize")
    if processing_mode == "identity" and (
        anonymization_plan is not None or regions_json is not None or detect_plates or detect_faces
    ):
        raise ClientInputError("Las regiones requieren seleccionar explícitamente anonymize")
    if processing_mode == "anonymize" and sum((
        anonymization_plan is not None, regions_json is not None,
        bool(detect_plates or detect_faces),
    )) != 1:
        raise ClientInputError(
            "Anonimización requiere un plan, JSON o detectores (--detect-faces y/o --detect-plates), sin mezclarlos"
        )
    print("Proyecto Delta - MPC MVP")
    print(f"\nVideo: {path.name}")

    if detect_plates:
        detector = LPDYuNet(confidence_threshold=plate_confidence_threshold)
        frame_detections: list[FrameDetections] = []

        def detect_sample(original_bgr, frame_index: int) -> None:
            frame = detector.detect_frame(original_bgr, frame_index)
            if not isinstance(frame, FrameDetections) or frame.frame_index != frame_index:
                raise ClientInputError("El detector devolvió un índice de fotograma incompatible")
            if not frame.succeeded:
                raise DetectorExecutionError(
                    f"Detector de placas fallido en frame_index={frame_index}: {frame.error}"
                )
            frame_detections.append(frame)

        processed = preprocessor.process(path, model, on_sample=detect_sample)
        if processed.source_size is None:
            raise ClientInputError("El preprocesador no informó las dimensiones originales")
        anonymization_plan = prepare_anonymization(
            frame_detections, source_size=processed.source_size,
            target_size=(processed.metadata.width, processed.metadata.height),
            frame_count=processed.metadata.frame_count,
        )
        print(f"[OK] Placas: {len(anonymization_plan.regions)} regiones preparadas; umbral {plate_confidence_threshold:g}")
        print("[AVISO] Solo se ocultan las regiones detectadas; el modelo puede omitir placas visibles.")
    else:
        processed = preprocessor.process(path, model)
    if detect_faces:
        target_size = (processed.metadata.width, processed.metadata.height)
        detections_by_frame: list[list[DetectedRegion]] = [
            [] for _ in range(processed.metadata.frame_count)
        ]
        # Las placas del plan y FaceRegion ya usan coordenadas MPC reales.
        # Unir aquí evita un segundo resize o volver a escalar las caras.
        if detect_plates:
            for region in anonymization_plan.regions:
                detections_by_frame[region.frame_index].append(DetectedRegion(
                    region.region_type, region.x, region.y, region.width, region.height,
                ))
        for region in processed.regions:
            if not 0 <= region.frame_id < processed.metadata.frame_count:
                raise ClientInputError(
                    f"YuNet devolvió frame_id fuera del payload: {region.frame_id}"
                )
            detections_by_frame[region.frame_id].append(DetectedRegion(
                RegionType.FACE, region.x, region.y, region.width, region.height,
            ))
        combined_results = tuple(
            FrameDetections.success(index, detections)
            for index, detections in enumerate(detections_by_frame)
        )
        anonymization_plan = prepare_anonymization(
            combined_results, source_size=target_size, target_size=target_size,
            frame_count=processed.metadata.frame_count,
        )
        print(f"[OK] YuNet: {len(processed.regions)} regiones faciales preparadas")
        print("[AVISO] Solo se ocultan los rostros detectados; YuNet puede omitir rostros visibles.")
    metadata = processed.metadata.as_dict()
    if processing_mode == "anonymize":
        if regions_json is not None:
            anonymization_plan = _plan_from_json(regions_json, processed.metadata)
        if not isinstance(anonymization_plan, AnonymizationPlan):
            raise ClientInputError("Se requiere un plan de anonimización validado")
        if not anonymization_plan.detections_complete:
            raise ClientInputError("Prepare el plan con resultados explícitos de todos los fotogramas")
        if (anonymization_plan.width, anonymization_plan.height, anonymization_plan.frame_count) != (
            processed.metadata.width, processed.metadata.height, processed.metadata.frame_count
        ) or len(processed.payload) != anonymization_plan.payload_size:
            raise ClientInputError("El plan no coincide con los fotogramas MPC realmente procesados")
        mode_detail = (
            "rostros y placas detectados" if detect_faces and detect_plates else
            "rostros detectados" if detect_faces else
            "placas detectadas" if detect_plates else
            "regiones proporcionadas explícitamente"
        )
        print(f"[MODO] ANONIMIZACIÓN MPC; {mode_detail}")
    else:
        print("[MODO] IDENTIDAD; no anonimiza")
    print("[OK] Video validado y preprocesado localmente")
    print(
        f"     {processed.metadata.frame_count} frames, "
        f"{processed.metadata.width}x{processed.metadata.height}, "
        f"{processed.metadata.fps:g} FPS, grayscale"
    )

    registration = coordinator.register_video(path.name, "mp4")
    print(f"[OK] Job registrado: {registration.id}")

    try:
        coordinator.prepare(registration.id, metadata)
        for endpoint in registration.parties:
            party_client.prepare_job(
                endpoint.id,
                endpoint.url,
                registration.id,
                metadata,
            )
            print(f"[OK] Party {endpoint.id} lista")
        coordinator.wait_for_status(registration.id, "MPC_READY")
        print("[OK] MPC listo")

        for endpoint in registration.parties:
            party_client.start_job(endpoint.id, endpoint.url, registration.id)
        coordinator.start(registration.id)

        if processing_mode == "anonymize":
            result = mpc_client.process_anonymize(processed.payload, anonymization_plan)
            anonymization_plan.validate_result(processed.payload, result)
        else:
            result = mpc_client.process_identity(processed.payload)
        print("[OK] Datos enviados y recibidos mediante MP-SPDZ ExternalIO")
        if processing_mode == "identity" and result != processed.payload:
            raise MpcResultMismatchError(
                "El resultado MPC no coincide exactamente con el video preprocesado"
            )

        destination = reconstructor.reconstruct(
            result,
            processed.metadata,
            registration.id,
            output_root,
        )
        print("[OK] Video reconstruido y validado")

        for endpoint in registration.parties:
            party_client.complete_job(endpoint.id, endpoint.url, registration.id)
        for endpoint in registration.parties:
            party_client.cleanup_job(endpoint.id, endpoint.url, registration.id)
        coordinator.complete(registration.id)
        coordinator.cleanup(registration.id)

        print(f"\nResultado:\n{destination}")
        print("\nEstado: COMPLETED")
        return 0
    except KeyboardInterrupt:
        _best_effort_failure_cleanup(
            registration,
            coordinator,
            party_client,
            "Operación cancelada por el usuario",
        )
        raise
    except Exception as exc:
        _best_effort_failure_cleanup(registration, coordinator, party_client, str(exc))
        raise


def _best_effort_failure_cleanup(
    registration: VideoRegistration,
    coordinator: CoordinatorClient,
    party_client: PartyClient,
    detail: str,
) -> None:
    for endpoint in registration.parties:
        try:
            party_client.fail_job(endpoint.id, endpoint.url, registration.id, detail)
        except (PartyClientError, ValueError):
            pass
        try:
            party_client.cleanup_job(endpoint.id, endpoint.url, registration.id)
        except (PartyClientError, ValueError):
            pass
    try:
        coordinator.fail(registration.id, detail)
    except (CoordinatorClientError, ValueError):
        pass
    try:
        coordinator.cleanup(registration.id)
    except (CoordinatorClientError, ValueError):
        pass


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command != "upload":
        return 2

    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    coordinator_url = os.getenv("COORDINATOR_URL", "http://localhost:8000")
    output_root = os.getenv("DELTA_OUTPUT_DIR", "output")
    try:
        return upload_video(

            video_path=args.video,
            coordinator=CoordinatorClient(coordinator_url),
            party_client=PartyClient(),
            preprocessor=VideoPreprocessor(
                max_width=args.mpc_max_width,
                max_height=args.mpc_max_height,
                target_fps=args.mpc_fps,
            ),
            mpc_client=MpcExternalClient(),
            reconstructor=VideoReconstructor(),
            output_root=output_root,
            processing_mode=args.processing_mode,
            regions_json=args.regions_json,
            detect_plates=args.detect_plates,
            detect_faces=args.detect_faces,
            plate_confidence_threshold=args.plate_confidence_threshold,
            model=YuNet(modelPath=DEFAULT_YUNET_MODEL,
                  inputSize=[320, 320],
                  confThreshold=args.conf_threshold,
                  nmsThreshold=args.nms_threshold,
                  topK=args.top_k,
                  backendId=backend_target_pairs[args.backend_target][0],
                  targetId=backend_target_pairs[args.backend_target][1],
                  )
        )
    except (
        ClientInputError,
        CoordinatorClientError,
        PartyClientError,
        VideoPreprocessingError,
        VideoReconstructionError,
        MpcExternalIOError,
        MpcResultMismatchError,
        DetectorExecutionError,
        PlateDetectionError,
        YuNetInitializationError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperación cancelada", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

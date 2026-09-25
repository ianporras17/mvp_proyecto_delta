"""CLI confiable para el MVP de video identidad con MP-SPDZ."""

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

from client.networking import (
    CoordinatorClient,
    CoordinatorClientError,
    PartyClient,
    PartyClientError,
    VideoRegistration,
)
from client.preprocessing import (
    VideoPreprocessingError,
    VideoPreprocessor,
    VideoReconstructionError,
    VideoReconstructor,
)
from mpc import MpcExternalClient, MpcExternalIOError


class ClientInputError(ValueError):
    """La entrada local no cumple el perfil del MVP."""


class MpcResultMismatchError(RuntimeError):
    """El cálculo identidad no devolvió exactamente los pixels enviados."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m client",
        description="Proyecto Delta - MVP MPC de video identidad.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    upload = commands.add_parser(
        "upload",
        help="Preprocesa un MP4, ejecuta MPC y reconstruye el resultado.",
    )
    upload.add_argument("video", type=Path, help="Ruta local al archivo .mp4")
    return parser


def _validated_video_path(path: Path) -> Path:
    if not path.exists():
        raise ClientInputError(f"El archivo no existe: {path}")
    if not path.is_file():
        raise ClientInputError(f"La ruta no corresponde a un archivo: {path}")
    if path.suffix.lower() != ".mp4":
        raise ClientInputError("El MVP solamente admite archivos .mp4")
    return path


def upload_video(
    video_path: Path,
    coordinator: CoordinatorClient,
    party_client: PartyClient,
    preprocessor: VideoPreprocessor,
    mpc_client: MpcExternalClient,
    reconstructor: VideoReconstructor,
    output_root: str | Path = "output",
) -> int:
    path = _validated_video_path(video_path)
    print("Proyecto Delta - MPC MVP")
    print(f"\nVideo: {path.name}")

    processed = preprocessor.process(path)
    metadata = processed.metadata.as_dict()
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

        result = mpc_client.process_identity(processed.payload)
        print("[OK] Datos enviados y recibidos mediante MP-SPDZ ExternalIO")
        if result != processed.payload:
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

    coordinator_url = os.getenv("COORDINATOR_URL", "http://localhost:8000")
    output_root = os.getenv("DELTA_OUTPUT_DIR", "output")
    try:
        return upload_video(
            video_path=args.video,
            coordinator=CoordinatorClient(coordinator_url),
            party_client=PartyClient(),
            preprocessor=VideoPreprocessor(),
            mpc_client=MpcExternalClient(),
            reconstructor=VideoReconstructor(),
            output_root=output_root,
        )
    except (
        ClientInputError,
        CoordinatorClientError,
        PartyClientError,
        VideoPreprocessingError,
        VideoReconstructionError,
        MpcExternalIOError,
        MpcResultMismatchError,
        ValueError,
    ) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperación cancelada", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

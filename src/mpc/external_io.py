"""Cliente del mecanismo oficial ExternalIO de MP-SPDZ."""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType
from typing import Iterator


CHUNK_SIZE = 4096
_IMPORT_LOCK = Lock()


class MpcExternalIOError(RuntimeError):
    """Falló la entrada privada o la revelación autorizada al cliente."""


@dataclass(frozen=True)
class MpcSettings:
    hostnames: tuple[str, str, str]
    port_base: int
    client_id: int
    mpspdz_root: Path
    certificate_dir: Path

    @classmethod
    def from_environment(cls) -> "MpcSettings":
        project_root = Path(__file__).resolve().parents[2]
        raw_hosts = os.getenv("MPC_PARTY_HOSTS", "localhost,localhost,localhost")
        hosts = tuple(item.strip() for item in raw_hosts.split(",") if item.strip())
        if len(hosts) != 3:
            raise ValueError("MPC_PARTY_HOSTS debe contener exactamente tres hosts")
        port_base = int(os.getenv("MPC_CLIENT_PORT_BASE", "14000"))
        client_id = int(os.getenv("MPC_CLIENT_ID", "0"))
        mpspdz_root = Path(
            os.getenv("MPSPDZ_ROOT", str(project_root.parent / "MP-SPDZ"))
        ).resolve()
        certificate_dir = Path(
            os.getenv(
                "MPC_CERT_DIR",
                str(project_root / ".runtime" / "Player-Data"),
            )
        ).resolve()
        if not 1 <= port_base <= 65533:
            raise ValueError("MPC_CLIENT_PORT_BASE debe permitir tres puertos TCP")
        if client_id < 0:
            raise ValueError("MPC_CLIENT_ID no puede ser negativo")
        return cls(
            hostnames=(hosts[0], hosts[1], hosts[2]),
            port_base=port_base,
            client_id=client_id,
            mpspdz_root=mpspdz_root,
            certificate_dir=certificate_dir,
        )


class MpcExternalClient:
    """Envía uint8 privados por bloques y recibe identidad solo en cliente."""

    def __init__(self, settings: MpcSettings | None = None) -> None:
        self._settings = settings or MpcSettings.from_environment()

    def process_identity(self, payload: bytes) -> bytes:
        if not payload:
            raise ValueError("El payload MPC no puede estar vacío")
        client_module = self._load_official_client()
        self._validate_certificates()
        number_chunks = math.ceil(len(payload) / CHUNK_SIZE)
        client = None
        try:
            with _working_directory(self._settings.certificate_dir.parent):
                client = client_module.Client(
                    list(self._settings.hostnames),
                    self._settings.port_base,
                    self._settings.client_id,
                )
            client.send_public_inputs([len(payload), number_chunks])
            result = bytearray()
            for offset in range(0, len(payload), CHUNK_SIZE):
                chunk = list(payload[offset : offset + CHUNK_SIZE])
                actual_length = len(chunk)
                if actual_length < CHUNK_SIZE:
                    chunk.extend([0] * (CHUNK_SIZE - actual_length))
                client.send_private_inputs(chunk)
                revealed = client.receive_outputs(CHUNK_SIZE)
                if len(revealed) != CHUNK_SIZE:
                    raise MpcExternalIOError("MP-SPDZ devolvió un bloque incompleto")
                for value in revealed[:actual_length]:
                    if not 0 <= value <= 255:
                        raise MpcExternalIOError(
                            f"MP-SPDZ devolvió un pixel fuera de uint8: {value}"
                        )
                    result.append(value)
            return bytes(result)
        except MpcExternalIOError:
            raise
        except (OSError, RuntimeError, ValueError, AssertionError) as exc:
            raise MpcExternalIOError(f"Falló ExternalIO de MP-SPDZ: {exc}") from exc
        finally:
            if client is not None:
                for connection in getattr(client, "sockets", []):
                    try:
                        connection.close()
                    except OSError:
                        pass

    def _load_official_client(self) -> ModuleType:
        external_io = self._settings.mpspdz_root / "ExternalIO"
        source = external_io / "client.py"
        if not source.is_file():
            raise MpcExternalIOError(f"No se encontró el cliente oficial: {source}")
        module_name = "_delta_mpspdz_external_client"
        existing = sys.modules.get(module_name)
        if existing is not None:
            return existing
        with _IMPORT_LOCK:
            existing = sys.modules.get(module_name)
            if existing is not None:
                return existing
            spec = importlib.util.spec_from_file_location(module_name, source)
            if spec is None or spec.loader is None:
                raise MpcExternalIOError("No se pudo cargar ExternalIO/client.py")
            module = importlib.util.module_from_spec(spec)
            sys.path.insert(0, str(external_io))
            try:
                spec.loader.exec_module(module)
            except Exception as exc:
                raise MpcExternalIOError(
                    f"No se pudo importar ExternalIO oficial: {exc}"
                ) from exc
            finally:
                try:
                    sys.path.remove(str(external_io))
                except ValueError:
                    pass
            sys.modules[module_name] = module
            return module

    def _validate_certificates(self) -> None:
        if self._settings.certificate_dir.name != "Player-Data":
            raise MpcExternalIOError(
                "MPC_CERT_DIR debe apuntar a un directorio llamado Player-Data"
            )
        required = [
            self._settings.certificate_dir / f"C{self._settings.client_id}.pem",
            self._settings.certificate_dir / f"C{self._settings.client_id}.key",
            *(self._settings.certificate_dir / f"P{i}.pem" for i in range(3)),
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise MpcExternalIOError(
                "Faltan certificados de desarrollo; ejecute docker compose up --build: "
                + ", ".join(missing)
            )


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """ExternalIO upstream busca Player-Data relativo al cwd."""

    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)

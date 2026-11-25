"""Cliente del mecanismo oficial ExternalIO de MP-SPDZ."""

from contextlib import contextmanager
from dataclasses import dataclass
import importlib.util
import math
import os
from pathlib import Path
import struct
import sys
from threading import Lock
from types import ModuleType
from typing import TYPE_CHECKING, Iterator
from time import perf_counter

import numpy as np

if TYPE_CHECKING:
    from client.anonymization import AnonymizationPlan


# Debe coincidir con el argumento de compilación del programa MPC (ver el .mpc).
CHUNK_SIZE = int(os.getenv("MPC_CHUNK_SIZE", "262144"))
ANONYMIZATION_ACK = (1145852993, 1)
HANDSHAKE_TIMEOUT_SECONDS = 10.0
# DELTA_EXTERNALIO_VECTORIZED=0 fuerza el cliente oficial valor por valor.
VECTORIZED_EXTERNALIO = os.getenv("DELTA_EXTERNALIO_VECTORIZED", "1") != "0"
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
        raw_hosts = os.getenv("MPC_PARTY_HOSTS", "127.0.0.1,127.0.0.1,127.0.0.1")
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
                str(project_root / ".runtime" / "mvp" / "Player-Data"),
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


class _VectorizedChannel:
    """Mismo protocolo que ExternalIO/client.py, con NumPy en lugar de un objeto
    Python por valor.

    Solo cubre Z2(64) semi-honesto (el programa usa ``program.active = False``):
    para cada mensaje los bytes en la red son idénticos a los del cliente oficial,
    así que no cambia nada en las parties. El modo activo, que envía tres valores
    por triple, no se soporta y se rechaza por longitud inesperada. Si el cliente
    no es el oficial real, ``create`` devuelve None y se usa el camino original.
    """

    def __init__(self, client) -> None:
        self._sockets = list(client.sockets)

    @classmethod
    def create(cls, client) -> "_VectorizedChannel | None":
        if not VECTORIZED_EXTERNALIO:
            return None
        sockets = getattr(client, "sockets", None)
        domain = getattr(client, "domain", None)
        clear_domain = getattr(client, "clear_domain", None)
        if (
            not sockets
            or getattr(domain, "modulus", None) != 1 << 64
            or getattr(clear_domain, "modulus", None) != 1 << 64
            or not all(hasattr(connection, "recv_into") for connection in sockets)
        ):
            return None
        return cls(client)

    @staticmethod
    def _receive_exactly(connection, count: int) -> bytearray:
        buffer = bytearray(count)
        view = memoryview(buffer)
        received = 0
        while received < count:
            read = connection.recv_into(view[received:], count - received)
            if not read:
                raise OSError("Conexión cerrada durante ExternalIO")
            received += read
        return buffer

    def _receive_sum(self, count: int) -> np.ndarray:
        """Suma módulo 2^64 de las tres respuestas: el equivalente de receive_triples."""
        total = np.zeros(count, dtype=np.uint64)
        for connection in self._sockets:
            (length,) = struct.unpack("<I", self._receive_exactly(connection, 4))
            if length != count * 8:
                raise MpcExternalIOError(
                    f"Longitud inesperada de ExternalIO: {length} bytes para {count} valores"
                )
            total += np.frombuffer(self._receive_exactly(connection, length), dtype="<u8")
        return total

    def _send_all(self, data: bytes) -> None:
        message = struct.pack("<i", len(data)) + data
        for connection in self._sockets:
            connection.sendall(message)

    def send_public_mask(self, mask: bytes) -> None:
        values = np.zeros(CHUNK_SIZE, dtype="<u8")
        values[: len(mask)] = np.frombuffer(mask, dtype=np.uint8)
        self._send_all(values.tobytes())

    def exchange(self, payload: bytes, offset: int, length: int) -> bytes:
        """Envía length píxeles privados y devuelve el bloque revelado al cliente."""
        values = np.zeros(CHUNK_SIZE, dtype=np.uint64)
        values[:length] = np.frombuffer(payload, dtype=np.uint8, count=length, offset=offset)
        values += self._receive_sum(CHUNK_SIZE)
        self._send_all(values.astype("<u8").tobytes())
        revealed = self._receive_sum(CHUNK_SIZE).view(np.int64)
        block = revealed[:length]
        if (block < 0).any() or (block > 255).any():
            bad = block[(block < 0) | (block > 255)][0]
            raise MpcExternalIOError(f"MP-SPDZ devolvió un pixel fuera de uint8: {int(bad)}")
        if revealed[length:].any():
            raise MpcExternalIOError("MP-SPDZ devolvió padding no nulo")
        return block.astype(np.uint8).tobytes()


class MpcExternalClient:
    """Envía uint8 privados por bloques; solo revela el resultado al cliente."""

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
            channel = _VectorizedChannel.create(client)
            started = perf_counter()
            for offset in range(0, len(payload), CHUNK_SIZE):
                if channel is not None:
                    result += channel.exchange(
                        payload, offset, min(CHUNK_SIZE, len(payload) - offset)
                    )
                else:
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
                if (offset // CHUNK_SIZE + 1) % 250 == 0 or offset + CHUNK_SIZE >= len(payload):
                    print(f"[PROGRESO] ExternalIO: {offset // CHUNK_SIZE + 1}/{number_chunks} bloques; {perf_counter()-started:.1f} s", flush=True)
            print(f"[TIEMPO] ExternalIO identidad: {perf_counter()-started:.3f} s", flush=True)
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

    def process_anonymize(self, payload: bytes, plan: "AnonymizationPlan") -> bytes:
        """Máscara pública y píxeles originales privados; nunca hay fallback.

        El plan debe prepararse con todos los resultados T024 antes de llamar.
        El header negativo selecciona el modo del nuevo circuito, sin cambiar
        el header de identidad anterior. Los tres ACK preceden cualquier píxel.
        """
        from client.anonymization import AnonymizationPlan

        if not isinstance(plan, AnonymizationPlan):
            raise ValueError("La anonimización requiere un AnonymizationPlan validado")
        if not plan.detections_complete:
            raise ValueError("Use prepare_anonymization con resultados de todos los fotogramas")
        if not isinstance(payload, bytes) or len(payload) != plan.payload_size:
            raise ValueError("Los píxeles no coinciden con las dimensiones del plan")
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
            client.send_public_inputs([-len(payload), number_chunks])
            if len(client.sockets) != 3:
                raise MpcExternalIOError("Se requieren ACK de las tres parties")
            # El lector upstream no detecta EOF en mitad de un cuerpo. Proteger
            # todas las lecturas, incluidas triples y salida, sin cambiar su MPC.
            client.sockets = [_HandshakeSocket(connection) for connection in client.sockets]
            for connection in client.sockets:
                previous_timeout = connection.gettimeout()
                try:
                    connection.settimeout(HANDSHAKE_TIMEOUT_SECONDS)
                    ack = client.receive_plain_values(socket=connection)
                finally:
                    connection.settimeout(previous_timeout)
                if tuple(ack) != ANONYMIZATION_ACK:
                    raise MpcExternalIOError("La party no confirmó el programa de anonimización")

            result = bytearray()
            channel = _VectorizedChannel.create(client)
            started = perf_counter()
            for offset in range(0, len(payload), CHUNK_SIZE):
                actual_length = min(CHUNK_SIZE, len(payload) - offset)
                if channel is not None:
                    channel.send_public_mask(plan.mask_chunk(offset, actual_length))
                    # Nunca se ocultan los píxeles en el cliente: enviar los originales.
                    result += channel.exchange(payload, offset, actual_length)
                else:
                    mask = list(plan.mask_chunk(offset, actual_length))
                    mask.extend([0] * (CHUNK_SIZE - actual_length))
                    client.send_public_inputs(mask)
                    # Nunca se ocultan los píxeles en el cliente: enviar los originales.
                    chunk = list(payload[offset : offset + actual_length])
                    chunk.extend([0] * (CHUNK_SIZE - actual_length))
                    client.send_private_inputs(chunk)
                    revealed = client.receive_outputs(CHUNK_SIZE)
                    if len(revealed) != CHUNK_SIZE:
                        raise MpcExternalIOError("MP-SPDZ devolvió un bloque incompleto")
                    for value in revealed[:actual_length]:
                        if not isinstance(value, int) or not 0 <= value <= 255:
                            raise MpcExternalIOError("MP-SPDZ devolvió un píxel fuera de uint8")
                        result.append(value)
                    if any(revealed[actual_length:]):
                        raise MpcExternalIOError("MP-SPDZ devolvió padding no nulo")
                if (offset // CHUNK_SIZE + 1) % 250 == 0 or offset + CHUNK_SIZE >= len(payload):
                    print(f"[PROGRESO] ExternalIO: {offset // CHUNK_SIZE + 1}/{number_chunks} bloques; {perf_counter()-started:.1f} s", flush=True)
            print(f"[TIEMPO] ExternalIO anonimización: {perf_counter()-started:.3f} s", flush=True)
            result = bytes(result)
            # Validar, no producir un resultado anonimizado localmente.
            plan.validate_result(payload, result)
            return result
        except MpcExternalIOError:
            raise
        except Exception as exc:
            raise MpcExternalIOError(f"Falló anonimización ExternalIO de MP-SPDZ: {exc}") from exc
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


class _HandshakeSocket:
    """Lecturas completas y EOF explícito durante todo el transporte anónimo."""

    def __init__(self, connection) -> None:
        self.connection = connection

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def recv(self, count: int) -> bytes:
        data = bytearray()
        while len(data) < count:
            part = self.connection.recv(count - len(data))
            if not part:
                raise OSError("Conexión cerrada durante ExternalIO de anonimización")
            data.extend(part)
        return bytes(data)


@contextmanager
def _working_directory(path: Path) -> Iterator[None]:
    """ExternalIO upstream busca Player-Data relativo al cwd."""

    original = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original)

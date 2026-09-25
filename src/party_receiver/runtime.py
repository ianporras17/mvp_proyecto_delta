"""Supervisor del proceso MP-SPDZ perteneciente a una party lógica."""

import os
from pathlib import Path
import subprocess
from threading import RLock
import time
from typing import BinaryIO


class MpcRuntimeError(RuntimeError):
    """El proceso replicated-ring no pudo iniciarse o terminó."""


def _environment_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class MpcRuntimeManager:
    """Mantiene un runtime MP-SPDZ persistente por contenedor."""

    def __init__(
        self,
        party_id: int,
        *,
        enabled: bool | None = None,
        root: str | Path | None = None,
        hosts_file: str | Path | None = None,
        program: str | None = None,
        startup_grace: float = 2.0,
    ) -> None:
        if party_id not in (0, 1, 2):
            raise ValueError("party_id debe ser 0, 1 o 2")
        self.party_id = party_id
        self.enabled = (
            _environment_flag("MPC_RUNTIME_ENABLED") if enabled is None else enabled
        )
        self.root = Path(root or os.getenv("MPSPDZ_ROOT", "/opt/MP-SPDZ"))
        self.hosts_file = Path(
            hosts_file or os.getenv("MPC_HOSTS_FILE", "/app/mpc/hosts-docker")
        )
        self.program = program or os.getenv("MPC_PROGRAM", "delta_video_identity")
        self.startup_grace = startup_grace
        self._process: subprocess.Popen[bytes] | None = None
        self._log_handle: BinaryIO | None = None
        self._lock = RLock()

    @property
    def binary(self) -> Path:
        return self.root / "replicated-ring-party.x"

    @property
    def log_path(self) -> Path:
        configured = Path(os.getenv("MPC_LOG_DIR", "/tmp/mvp-proyecto-delta"))
        return configured / f"party-{self.party_id}.log"

    def start(self) -> None:
        if not self.enabled:
            return
        with self._lock:
            if self._process is not None and self._process.poll() is None:
                return
            # ExternalIO finaliza el proceso si pasa su timeout sin clientes.
            # Un prepare posterior debe poder crear un proceso limpio de nuevo.
            self._process = None
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            if not self.binary.is_file():
                raise MpcRuntimeError(f"No existe el runtime MP-SPDZ: {self.binary}")
            if not self.hosts_file.is_file():
                raise MpcRuntimeError(f"No existe el archivo de hosts: {self.hosts_file}")
            schedule = self.root / "Programs" / "Schedules" / f"{self.program}.sch"
            if not schedule.is_file():
                raise MpcRuntimeError(f"No existe el programa compilado: {schedule}")
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_handle = self.log_path.open("ab", buffering=0)
            command = [
                str(self.binary),
                str(self.party_id),
                self.program,
                "-ip",
                str(self.hosts_file),
            ]
            self._process = subprocess.Popen(
                command,
                cwd=self.root,
                stdin=subprocess.DEVNULL,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
            )

        deadline = time.monotonic() + self.startup_grace
        while time.monotonic() < deadline:
            if not self.is_running:
                raise MpcRuntimeError(
                    f"Party {self.party_id} terminó al iniciar; revise {self.log_path}"
                )
            time.sleep(0.05)

    @property
    def is_running(self) -> bool:
        if not self.enabled:
            return True
        process = self._process
        return process is not None and process.poll() is None

    @property
    def status(self) -> str:
        if not self.enabled:
            return "disabled"
        return "running" if self.is_running else "stopped"

    def require_running(self) -> None:
        if not self.is_running:
            raise MpcRuntimeError(
                f"El runtime MP-SPDZ de Party {self.party_id} no está ejecutándose"
            )

    def stop(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None

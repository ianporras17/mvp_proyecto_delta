"""Chunking del adaptador; la criptografía permanece en MP-SPDZ upstream."""

from pathlib import Path
from types import SimpleNamespace

from mpc.external_io import CHUNK_SIZE, MpcExternalClient, MpcSettings


class FakeOfficialClient:
    instance: "FakeOfficialClient"

    def __init__(self, hostnames: list[str], port_base: int, client_id: int) -> None:
        self.hostnames = hostnames
        self.port_base = port_base
        self.client_id = client_id
        self.public: list[int] = []
        self.pending: list[int] = []
        self.private_lengths: list[int] = []
        self.sockets: list = []
        FakeOfficialClient.instance = self

    def send_public_inputs(self, values: list[int]) -> None:
        self.public = values

    def send_private_inputs(self, values: list[int]) -> None:
        self.pending = values
        self.private_lengths.append(len(values))

    def receive_outputs(self, count: int) -> list[int]:
        assert count == CHUNK_SIZE
        return self.pending


def test_processes_and_reassembles_fixed_size_chunks(
    tmp_path: Path, monkeypatch
) -> None:
    settings = MpcSettings(
        ("p0", "p1", "p2"),
        14000,
        0,
        tmp_path / "MP-SPDZ",
        tmp_path / "Player-Data",
    )
    adapter = MpcExternalClient(settings)
    monkeypatch.setattr(
        adapter,
        "_load_official_client",
        lambda: SimpleNamespace(Client=FakeOfficialClient),
    )
    monkeypatch.setattr(adapter, "_validate_certificates", lambda: None)
    payload = bytes(range(256)) * (CHUNK_SIZE // 256 + 1)

    result = adapter.process_identity(payload)

    assert result == payload
    assert FakeOfficialClient.instance.public == [len(payload), 2]
    assert FakeOfficialClient.instance.private_lengths == [CHUNK_SIZE, CHUNK_SIZE]

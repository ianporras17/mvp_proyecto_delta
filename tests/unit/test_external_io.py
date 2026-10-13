"""Chunking del adaptador; la criptografía permanece en MP-SPDZ upstream."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from mpc.external_io import (
    CHUNK_SIZE,
    MpcExternalClient,
    MpcExternalIOError,
    MpcSettings,
)


class FakeSocket:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class FakeOfficialClient:
    instance: "FakeOfficialClient"

    def __init__(self, hostnames: list[str], port_base: int, client_id: int) -> None:
        self.hostnames = hostnames
        self.port_base = port_base
        self.client_id = client_id
        self.public: list[int] = []
        self.pending: list[int] = []
        self.private_lengths: list[int] = []
        self.private_inputs: list[list[int]] = []
        self.sockets = [FakeSocket() for _ in range(3)]
        FakeOfficialClient.instance = self

    def send_public_inputs(self, values: list[int]) -> None:
        self.public = values

    def send_private_inputs(self, values: list[int]) -> None:
        self.pending = values
        self.private_lengths.append(len(values))
        self.private_inputs.append(values.copy())

    def receive_outputs(self, count: int) -> list[int]:
        assert count == CHUNK_SIZE
        return self.pending


@pytest.fixture
def adapter(
    tmp_path: Path, monkeypatch
) -> MpcExternalClient:
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
    return adapter


def test_processes_and_reassembles_fixed_size_chunks(
    adapter: MpcExternalClient,
) -> None:
    payload = bytes(range(256)) * (CHUNK_SIZE // 256 + 1)

    result = adapter.process_identity(payload)

    assert result == payload
    assert FakeOfficialClient.instance.public == [len(payload), 2]
    assert FakeOfficialClient.instance.private_lengths == [CHUNK_SIZE, CHUNK_SIZE]
    assert all(connection.closed for connection in FakeOfficialClient.instance.sockets)


@pytest.mark.parametrize(
    "length", [1, CHUNK_SIZE - 1, CHUNK_SIZE, CHUNK_SIZE + 1, 3 * CHUNK_SIZE + 7]
)
def test_preserves_byte_order_and_zero_padding_at_chunk_boundaries(
    adapter: MpcExternalClient, length: int,
) -> None:
    payload = (bytes(range(256)) * ((length + 255) // 256))[:length]
    assert adapter.process_identity(payload) == payload

    client = FakeOfficialClient.instance
    number_chunks = (length + CHUNK_SIZE - 1) // CHUNK_SIZE
    assert client.public == [length, number_chunks]
    assert client.private_lengths == [CHUNK_SIZE] * number_chunks
    transmitted = [value for chunk in client.private_inputs for value in chunk]
    assert transmitted[:length] == list(payload)
    assert transmitted[length:] == [0] * (number_chunks * CHUNK_SIZE - length)
    assert all(connection.closed for connection in client.sockets)


def test_incomplete_output_is_not_treated_as_success(
    adapter: MpcExternalClient, monkeypatch,
) -> None:
    monkeypatch.setattr(
        FakeOfficialClient, "receive_outputs", lambda self, count: [0] * (count - 1)
    )

    with pytest.raises(MpcExternalIOError, match="bloque incompleto"):
        adapter.process_identity(b"\x00\xff")

    assert all(connection.closed for connection in FakeOfficialClient.instance.sockets)


@pytest.mark.parametrize("invalid_value", [-1, 256])
def test_out_of_uint8_output_is_not_treated_as_success(
    adapter: MpcExternalClient, monkeypatch, invalid_value: int,
) -> None:
    monkeypatch.setattr(
        FakeOfficialClient,
        "receive_outputs",
        lambda self, count: [invalid_value] + [0] * (count - 1),
    )

    with pytest.raises(MpcExternalIOError, match="fuera de uint8"):
        adapter.process_identity(b"\x00\xff")

    assert all(connection.closed for connection in FakeOfficialClient.instance.sockets)

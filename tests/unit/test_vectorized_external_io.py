"""El canal NumPy debe ser equivalente, byte a byte, al ExternalIO/client.py oficial."""

import socket
import struct
from threading import Thread

import numpy as np
import pytest

import mpc.external_io as external_io
from mpc.external_io import (
    MpcExternalClient, MpcExternalIOError, MpcSettings, _VectorizedChannel,
)

# El protocolo no depende del tamaño del bloque; uno pequeño evita mensajes de
# megabytes entre los hilos del test.
CHUNK_SIZE = 4096


@pytest.fixture(autouse=True)
def small_chunks(monkeypatch):
    monkeypatch.setattr(external_io, "CHUNK_SIZE", CHUNK_SIZE)


def _message(values: np.ndarray) -> bytes:
    data = np.asarray(values, dtype="<u8").tobytes()
    return struct.pack("<i", len(data)) + data


def _receive_exactly(connection: socket.socket, count: int) -> bytes:
    data = bytearray()
    while len(data) < count:
        part = connection.recv(count - len(data))
        assert part, "el cliente cerró la conexión"
        data.extend(part)
    return bytes(data)


def _receive_message(connection: socket.socket) -> bytes:
    (length,) = struct.unpack("<i", _receive_exactly(connection, 4))
    return _receive_exactly(connection, length)


class _Parties:
    """Tres sockets conectados: el cliente usa un extremo y el test simula a las parties."""

    def __init__(self) -> None:
        pairs = [socket.socketpair() for _ in range(3)]
        self.client = [pair[0] for pair in pairs]
        self.party = [pair[1] for pair in pairs]

    def close(self) -> None:
        for connection in (*self.client, *self.party):
            connection.close()


@pytest.fixture
def parties():
    state = _Parties()
    yield state
    state.close()


@pytest.fixture(scope="module")
def official_module():
    try:
        return MpcExternalClient(MpcSettings.from_environment())._load_official_client()
    except Exception as exc:  # noqa: BLE001 - sin el clon de MP-SPDZ no hay referencia
        pytest.skip(f"No se pudo cargar ExternalIO/client.py oficial: {exc}")


def _client(module, sockets):
    client = object.__new__(module.Client)
    client.sockets = sockets
    client.domain = module.Z2(64)
    client.clear_domain = module.Z2(64)
    return client


def _random_shares(seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 2**64, CHUNK_SIZE, dtype=np.uint64) for _ in range(3)]


def _send_all(parties: _Parties, shares: list[np.ndarray]) -> None:
    for connection, values in zip(parties.party, shares):
        connection.sendall(_message(values))


def test_receive_matches_official_including_values_that_wrap_modulo_2_64(parties, official_module):
    shares = _random_shares(1)
    _send_all(parties, shares)
    channel = _VectorizedChannel(_client(official_module, parties.client))

    revealed = channel._receive_sum(CHUNK_SIZE).view(np.int64)

    reference = _Parties()
    try:
        _send_all(reference, shares)
        official = _client(official_module, reference.client).receive_outputs(CHUNK_SIZE)
    finally:
        reference.close()
    assert revealed.tolist() == official
    assert (revealed < 0).any(), "la muestra debe cubrir valores con signo"


def test_private_input_message_is_byte_identical_to_official_client(parties, official_module):
    pixels = bytes(range(256)) * 8
    shares = _random_shares(2)
    _send_all(parties, shares)
    channel = _VectorizedChannel(_client(official_module, parties.client))
    sent: list[bytes] = []

    def parties_side() -> None:
        # Tras enviar los triples, cada party recibe un único mensaje con la entrada enmascarada.
        for connection in parties.party:
            sent.append(_receive_message(connection))
        _send_all(parties, _random_shares(3))

    thread = Thread(target=parties_side)
    thread.start()
    try:
        channel.exchange(pixels, 0, len(pixels))
    except MpcExternalIOError:
        pass  # las respuestas aleatorias no son uint8 válidos; solo importa lo enviado
    thread.join(timeout=10)

    reference = _Parties()
    try:
        _send_all(reference, shares)
        official = _client(official_module, reference.client)
        official.send_private_inputs(list(pixels) + [0] * (CHUNK_SIZE - len(pixels)))
        expected = [_receive_message(connection) for connection in reference.party]
    finally:
        reference.close()
    assert sent == expected
    assert len(set(sent)) == 1


def test_public_mask_message_is_byte_identical_to_official_client(parties, official_module):
    mask = bytes([0, 1, 1, 0, 1]) * 40
    channel = _VectorizedChannel(_client(official_module, parties.client))
    channel.send_public_mask(mask)
    sent = [_receive_message(connection) for connection in parties.party]

    reference = _Parties()
    try:
        _client(official_module, reference.client).send_public_inputs(
            list(mask) + [0] * (CHUNK_SIZE - len(mask))
        )
        expected = [_receive_message(connection) for connection in reference.party]
    finally:
        reference.close()
    assert sent == expected


def test_exchange_returns_revealed_bytes_and_ignores_zero_padding(parties, official_module):
    pixels = bytes(range(200))
    channel = _VectorizedChannel(_client(official_module, parties.client))
    triples = _random_shares(4)
    answer = np.zeros(CHUNK_SIZE, dtype=np.uint64)
    answer[: len(pixels)] = np.frombuffer(pixels, dtype=np.uint8)[::-1]
    split = [_random_shares(5)[0], _random_shares(6)[0]]
    split.append(answer - split[0] - split[1])  # las tres respuestas suman la salida

    def parties_side() -> None:
        _send_all(parties, triples)
        for connection in parties.party:
            _receive_message(connection)
        _send_all(parties, split)

    thread = Thread(target=parties_side)
    thread.start()
    block = channel.exchange(pixels, 0, len(pixels))
    thread.join(timeout=10)

    assert block == pixels[::-1]


@pytest.mark.parametrize("bad", [256, 2**64 - 1, 2**63])
def test_pixel_outside_uint8_is_rejected(parties, official_module, bad):
    channel = _VectorizedChannel(_client(official_module, parties.client))
    answer = np.zeros(CHUNK_SIZE, dtype=np.uint64)
    answer[3] = bad
    split = [_random_shares(7)[0], _random_shares(8)[0]]
    split.append(answer - split[0] - split[1])

    def parties_side() -> None:
        _send_all(parties, _random_shares(9))
        for connection in parties.party:
            _receive_message(connection)
        _send_all(parties, split)

    thread = Thread(target=parties_side)
    thread.start()
    with pytest.raises(MpcExternalIOError, match="fuera de uint8"):
        channel.exchange(bytes(10), 0, 10)
    thread.join(timeout=10)


def test_nonzero_padding_is_rejected(parties, official_module):
    channel = _VectorizedChannel(_client(official_module, parties.client))
    answer = np.zeros(CHUNK_SIZE, dtype=np.uint64)
    answer[50] = 7
    split = [_random_shares(10)[0], _random_shares(11)[0]]
    split.append(answer - split[0] - split[1])

    def parties_side() -> None:
        _send_all(parties, _random_shares(12))
        for connection in parties.party:
            _receive_message(connection)
        _send_all(parties, split)

    thread = Thread(target=parties_side)
    thread.start()
    with pytest.raises(MpcExternalIOError, match="padding"):
        channel.exchange(bytes(10), 0, 10)
    thread.join(timeout=10)


def test_unexpected_length_such_as_active_mode_is_rejected(parties, official_module):
    channel = _VectorizedChannel(_client(official_module, parties.client))
    for connection in parties.party:
        connection.sendall(_message(np.zeros(CHUNK_SIZE * 3, dtype=np.uint64)))
    with pytest.raises(MpcExternalIOError, match="Longitud inesperada"):
        channel._receive_sum(CHUNK_SIZE)


def test_closed_connection_is_an_error_not_an_endless_wait(parties, official_module):
    channel = _VectorizedChannel(_client(official_module, parties.client))
    parties.party[0].sendall(struct.pack("<i", CHUNK_SIZE * 8) + b"\x00" * 100)
    parties.party[0].close()
    with pytest.raises(OSError, match="Conexi"):
        channel._receive_sum(CHUNK_SIZE)


def test_create_declines_clients_that_are_not_the_official_z2_64(official_module):
    class Fake:
        sockets = [object()]

    assert _VectorizedChannel.create(Fake()) is None

    class WrongDomain:
        sockets = [socket.socket()]
        domain = type("D", (), {"modulus": 2**128})
        clear_domain = type("D", (), {"modulus": 2**64})

    try:
        assert _VectorizedChannel.create(WrongDomain()) is None
    finally:
        WrongDomain.sockets[0].close()

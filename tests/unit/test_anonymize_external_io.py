"""Anonymization wire ordering and fail-closed behavior, with unit fake sockets."""

from pathlib import Path
import struct
from types import SimpleNamespace

import pytest

from client.anonymization import AnonymizationPlan, prepare_anonymization
from client.regions import DetectedRegion, FrameDetections, SensitiveRegion
from mpc.external_io import (
    ANONYMIZATION_ACK, CHUNK_SIZE, HANDSHAKE_TIMEOUT_SECONDS,
    MpcExternalClient, MpcExternalIOError, MpcSettings, _HandshakeSocket,
)


def _prepared(width, height, frame_count, regions=()):
    results = (
        FrameDetections.success(index, (
            DetectedRegion(region.region_type, region.x, region.y, region.width, region.height)
            for region in regions if region.frame_index == index
        )) for index in range(frame_count)
    )
    return prepare_anonymization(
        results, source_size=(width, height), target_size=(width, height), frame_count=frame_count,
    )


class FakeSocket:
    def __init__(self, index):
        self.index, self.closed, self.timeout = index, False, 17.0
        self.timeouts = []
        self.reads = 0

    def gettimeout(self):
        return self.timeout

    def settimeout(self, value):
        self.timeouts.append(value)
        self.timeout = value

    def recv(self, count):
        self.reads += 1
        return b"\x01" if self.reads == 1 else b""

    def close(self):
        self.closed = True


@pytest.fixture
def adapter_factory(tmp_path: Path, monkeypatch):
    def create(fault=None, socket_count=3):
        state = SimpleNamespace(instance=None, load_calls=0, cert_calls=0)

        class FakeOfficialClient:
            def __init__(self, hosts, port, client_id):
                self.sockets = [FakeSocket(i) for i in range(socket_count)]
                self.events, self.masks, self.private = [], [], []
                state.instance = self

            def send_public_inputs(self, values):
                name = "header" if not self.events else "mask"
                self.events.append((name, list(values)))
                if name == "mask":
                    self.masks.append(list(values))

            def receive_plain_values(self, socket):
                connection = socket.connection
                self.events.append(("ack", connection.index))
                if fault == "ack_timeout" and connection.index == 1:
                    raise TimeoutError("ACK timeout")
                if fault == "ack_eof" and connection.index == 1:
                    while socket.recv(4):
                        pass  # Proxy must reject the EOF following a partial read.
                if fault == "ack_wrong" and connection.index == 1:
                    return [ANONYMIZATION_ACK[0], 0]
                return list(ANONYMIZATION_ACK)

            def send_private_inputs(self, values):
                self.events.append(("private", len(values)))
                self.private.append(list(values))

            def receive_outputs(self, count):
                assert count == CHUNK_SIZE
                self.events.append(("output", count))
                output = [0 if flag else value for flag, value in zip(self.masks[-1], self.private[-1])]
                if fault == "output_disconnect" and len(self.private) == 2:
                    raise OSError("socket closed during output")
                if fault == "short_output":
                    return output[:-1]
                if fault in ("negative", "overflow", "noninteger"):
                    output[0] = {"negative": -1, "overflow": 256, "noninteger": 1.5}[fault]
                if fault == "outside_changed":
                    output[0] = 0
                if fault == "inside_visible" and len(self.private) == 1:
                    output[1] = 99
                if fault == "bad_padding" and len(self.private) == 2:
                    output[-1] = 1
                return output

        settings = MpcSettings(("p0", "p1", "p2"), 14000, 0, tmp_path / "MP-SPDZ", tmp_path / "Player-Data")
        adapter = MpcExternalClient(settings)

        def load():
            state.load_calls += 1
            return SimpleNamespace(Client=FakeOfficialClient)

        def certificates():
            state.cert_calls += 1

        monkeypatch.setattr(adapter, "_load_official_client", load)
        monkeypatch.setattr(adapter, "_validate_certificates", certificates)
        monkeypatch.setattr(adapter, "process_identity", lambda _: pytest.fail("No identity fallback allowed"))
        return adapter, state

    return create


def test_header_ack_public_mask_original_private_pixels_padding_and_chunk_alignment(adapter_factory):
    plan = _prepared(101, 41, 2, (
        SensitiveRegion(0, "face", 90, 40, 11, 1),
        SensitiveRegion(1, "plate", 0, 0, 15, 2),
    ))
    payload = bytes((index % 255) + 1 for index in range(plan.payload_size))
    adapter, state = adapter_factory()

    result = adapter.process_anonymize(payload, plan)

    client = state.instance
    assert client.events[0] == ("header", [-len(payload), 3])
    assert client.events[1:4] == [("ack", 0), ("ack", 1), ("ack", 2)]
    assert [name for name, _ in client.events[4:]] == ["mask", "private", "output"] * 3
    for index, (mask, private) in enumerate(zip(client.masks, client.private)):
        offset = index * CHUNK_SIZE
        length = min(CHUNK_SIZE, len(payload) - offset)
        assert len(mask) == len(private) == CHUNK_SIZE
        assert mask[:length] == list(plan.mask_chunk(offset, length))
        assert private[:length] == list(payload[offset:offset + length])
        assert mask[length:] == private[length:] == [0] * (CHUNK_SIZE - length)
    assert all(socket.closed for socket in client.sockets)
    assert all(socket.timeouts == [HANDSHAKE_TIMEOUT_SECONDS, 17.0] for socket in client.sockets)
    assert payload[4130] != 0 and result[4130] == 0
    plan.validate_result(payload, result)


def test_empty_regions_still_use_anonymization_header_and_three_acks(adapter_factory):
    adapter, state = adapter_factory()
    payload = bytes(range(16))
    assert adapter.process_anonymize(payload, _prepared(4, 4, 1)) == payload
    assert state.instance.events[0] == ("header", [-16, 1])
    assert not any(state.instance.masks[0])


@pytest.mark.parametrize("fault", ["ack_wrong", "ack_timeout", "ack_eof"])
def test_failed_or_unsupported_ack_sends_no_private_pixels_and_closes_all_sockets(adapter_factory, fault):
    adapter, state = adapter_factory(fault)
    with pytest.raises(MpcExternalIOError):
        adapter.process_anonymize(bytes(12), _prepared(4, 3, 1))
    assert state.instance.private == state.instance.masks == []
    assert all(socket.closed for socket in state.instance.sockets)
    assert state.instance.sockets[1].timeout == 17.0


@pytest.mark.parametrize("socket_count", [0, 1, 2, 4])
def test_requires_exactly_three_party_ack_connections(adapter_factory, socket_count):
    adapter, state = adapter_factory(socket_count=socket_count)
    with pytest.raises(MpcExternalIOError):
        adapter.process_anonymize(bytes(12), _prepared(4, 3, 1))
    assert state.instance.private == []
    assert all(socket.closed for socket in state.instance.sockets)


@pytest.mark.parametrize("fault", ["output_disconnect", "short_output", "negative", "overflow", "noninteger", "outside_changed", "inside_visible", "bad_padding"])
def test_bad_output_aborts_without_identity_fallback_and_always_closes_connections(adapter_factory, fault):
    adapter, state = adapter_factory(fault)
    plan = _prepared(101, 41, 1, (SensitiveRegion(0, "face", 1, 0, 3, 1),))
    with pytest.raises(MpcExternalIOError):
        adapter.process_anonymize(bytes([77]) * plan.payload_size, plan)
    assert len(state.instance.private) >= 1
    assert all(socket.closed for socket in state.instance.sockets)


@pytest.mark.parametrize(("payload", "plan"), [(bytes(12), None), (bytes(12), object()), (bytes(11), _prepared(4, 3, 1)), (b"", _prepared(4, 3, 1)), (bytearray(12), _prepared(4, 3, 1))])
def test_invalid_plan_or_payload_is_rejected_before_import_certificates_or_connections(adapter_factory, payload, plan):
    adapter, state = adapter_factory()
    with pytest.raises(ValueError):
        adapter.process_anonymize(payload, plan)
    assert state.load_calls == state.cert_calls == 0
    assert state.instance is None


@pytest.mark.parametrize("regions", [(), (SensitiveRegion(0, "face", 1, 0, 3, 1),)])
def test_manual_geometry_plan_never_proves_detector_executed_and_is_rejected(adapter_factory, regions):
    manual = AnonymizationPlan(4, 3, 1, regions)
    prepared = _prepared(4, 3, 1, regions)
    assert manual.detections_complete is False
    assert prepared.detections_complete is True
    adapter, state = adapter_factory()
    with pytest.raises(ValueError):
        adapter.process_anonymize(bytes(12), manual)
    assert state.load_calls == state.cert_calls == 0
    assert state.instance is None


@pytest.mark.parametrize("body", [b"", b"abc", b"public ACK or output data"])
def test_socket_proxy_handles_partial_upstream_frame_headers_and_bodies(tmp_path, body):
    root = Path(__file__).resolve().parents[3] / "MP-SPDZ"
    settings = MpcSettings(("p0", "p1", "p2"), 14000, 0, root, tmp_path / "Player-Data")
    module = MpcExternalClient(settings)._load_official_client()
    wire = struct.pack("<I", len(body)) + body

    class BytewiseSocket:
        index = 7

        def recv(self, count):
            nonlocal wire
            value, wire = wire[:1], wire[1:]
            return value

    wrapped = _HandshakeSocket(BytewiseSocket())
    stream = module.octetStream()
    stream.Receive(wrapped)
    assert stream.buf == body
    assert wrapped.index == 7  # Attribute delegation remains upstream compatible.
    assert wire == b""


@pytest.mark.parametrize("truncated", [b"\x05", struct.pack("<I", 5) + b"ab"])
def test_partial_upstream_frame_eof_is_an_error_not_an_endless_receive_loop(tmp_path, truncated):
    root = Path(__file__).resolve().parents[3] / "MP-SPDZ"
    settings = MpcSettings(("p0", "p1", "p2"), 14000, 0, root, tmp_path / "Player-Data")
    module = MpcExternalClient(settings)._load_official_client()

    class TruncatedSocket:
        def recv(self, count):
            nonlocal truncated
            value, truncated = truncated[:count], truncated[count:]
            return value

    with pytest.raises(OSError, match="Conexi"):
        module.octetStream().Receive(_HandshakeSocket(TruncatedSocket()))

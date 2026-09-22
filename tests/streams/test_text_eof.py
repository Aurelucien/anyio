from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from anyio import (
    BrokenResourceError,
    CancelScope,
    ClosedResourceError,
    EndOfStream,
    create_memory_object_stream,
)
from anyio.abc import ByteReceiveStream
from anyio.streams.stapled import StapledObjectStream
from anyio.streams.text import TextReceiveStream, TextStream


@pytest.mark.parametrize("stream_class", [TextReceiveStream, TextStream])
@pytest.mark.parametrize("errors", ["strict", "replace", "ignore"])
@pytest.mark.parametrize("encoding,data", [("utf-8", b"\xc3"), ("utf-16-le", b"a")])
async def test_incomplete_character_at_eof(
    stream_class: type[TextReceiveStream | TextStream],
    errors: str,
    encoding: str,
    data: bytes,
) -> None:
    send, receive = create_memory_object_stream[bytes](1)
    async with send, receive:
        await send.send(data)
        await send.aclose()
        transport = StapledObjectStream(send, receive)
        stream = stream_class(transport, encoding=encoding, errors=errors)
        if errors == "strict":
            with pytest.raises(UnicodeDecodeError):
                await stream.receive()
        elif errors == "replace":
            assert await stream.receive() == "\ufffd"
        else:
            with pytest.raises(EndOfStream):
                await stream.receive()

        with pytest.raises(EndOfStream):
            await stream.receive()


async def test_complete_character_split_across_chunks() -> None:
    send, receive = create_memory_object_stream[bytes](2)
    async with send, receive:
        await send.send(b"\xc3")
        await send.send(b"\xa9")
        await send.aclose()
        stream = TextReceiveStream(receive)
        assert await stream.receive() == "\u00e9"
        with pytest.raises(EndOfStream):
            await stream.receive()


@pytest.mark.parametrize("data", [b"", b"complete", b"\xc3\xa9"])
async def test_complete_input_at_eof(data: bytes) -> None:
    send, receive = create_memory_object_stream[bytes](1)
    async with send, receive:
        await send.send(data)
        await send.aclose()
        stream = TextReceiveStream(receive)
        assert "".join([chunk async for chunk in stream]) == data.decode()
        with pytest.raises(EndOfStream):
            await stream.receive()


@pytest.mark.parametrize("reach_eof", [False, True])
async def test_close_with_incomplete_character(reach_eof: bool) -> None:
    send, receive = create_memory_object_stream[bytes](1)
    async with send, receive:
        await send.send(b"a\xc3")
        stream = TextReceiveStream(receive, errors="replace")
        assert await stream.receive() == "a"
        if reach_eof:
            await send.aclose()
            assert await stream.receive() == "\ufffd"

        await stream.aclose()
        with pytest.raises(ClosedResourceError):
            await stream.receive()


async def test_transport_closed_after_eof() -> None:
    send, receive = create_memory_object_stream[bytes](1)
    async with send, receive:
        await send.aclose()
        stream = TextReceiveStream(receive)
        with pytest.raises(EndOfStream):
            await stream.receive()

        await receive.aclose()
        with pytest.raises(ClosedResourceError):
            await stream.receive()


async def test_broken_transport_does_not_finalize(mocker: MockerFixture) -> None:
    transport = mocker.Mock(spec=ByteReceiveStream)
    transport.receive = mocker.AsyncMock(
        side_effect=[b"a\xc3", BrokenResourceError, b"\xa9", EndOfStream]
    )
    stream = TextReceiveStream(transport)
    assert await stream.receive() == "a"
    with pytest.raises(BrokenResourceError):
        await stream.receive()

    assert await stream.receive() == "\u00e9"
    with pytest.raises(EndOfStream):
        await stream.receive()


async def test_cancelled_receive_does_not_finalize() -> None:
    send, receive = create_memory_object_stream[bytes](1)
    async with send, receive:
        await send.send(b"a\xc3")
        stream = TextReceiveStream(receive)
        assert await stream.receive() == "a"
        with CancelScope() as scope:
            scope.cancel()
            await stream.receive()

        assert scope.cancelled_caught
        await send.send(b"\xa9")
        assert await stream.receive() == "\u00e9"

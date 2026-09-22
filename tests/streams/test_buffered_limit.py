from __future__ import annotations

import pytest

from anyio import DelimiterNotFound, create_memory_object_stream
from anyio.streams.buffered import BufferedByteReceiveStream, BufferedByteStream
from anyio.streams.stapled import StapledObjectStream


@pytest.mark.parametrize(
    "stream_class", [BufferedByteReceiveStream, BufferedByteStream]
)
@pytest.mark.parametrize("layout", ["one-chunk", "split", "buffered"])
@pytest.mark.parametrize(
    "data,delimiter,limit",
    [(b"abcde!", b"!", 3), (b"abc\r\n", b"\r\n", 4)],
)
async def test_delimiter_beyond_limit(
    stream_class: type[BufferedByteReceiveStream | BufferedByteStream],
    layout: str,
    data: bytes,
    delimiter: bytes,
    limit: int,
) -> None:
    send, receive = create_memory_object_stream[bytes](2)
    async with send, receive:
        stream = stream_class(StapledObjectStream(send, receive))
        if layout == "buffered":
            stream.feed_data(data)
        elif layout == "split":
            await send.send(data[:limit])
            await send.send(data[limit:])
        else:
            await send.send(data)

        await send.aclose()
        with pytest.raises(DelimiterNotFound):
            await stream.receive_until(delimiter, limit)

        # The rejected read must not discard data, including a partial delimiter.
        assert (
            await stream.receive_until(delimiter, len(data)) == data[: -len(delimiter)]
        )
        assert stream.buffer == b""


@pytest.mark.parametrize("split", [1, 3, 4, 5])
async def test_delimiter_ending_at_limit(split: int) -> None:
    data = b"abc\r\nrest"
    send, receive = create_memory_object_stream[bytes](2)
    async with send, receive:
        await send.send(data[:split])
        await send.send(data[split:])
        await send.aclose()
        stream = BufferedByteReceiveStream(receive)

        assert await stream.receive_until(b"\r\n", 5) == b"abc"
        assert await stream.receive_exactly(4) == b"rest"

"""Bit-level I/O streams for arithmetic coding."""

from __future__ import annotations

import io


class BitOutputStream:
    """Writes individual bits to a byte stream."""

    def __init__(self, out: io.BytesIO) -> None:
        self._out = out
        self._buffer = 0
        self._bits_in_buffer = 0
        self._closed = False

    def write_bit(self, bit: int) -> None:
        if self._closed:
            raise ValueError("stream is closed")
        self._buffer = (self._buffer << 1) | (bit & 1)
        self._bits_in_buffer += 1
        if self._bits_in_buffer == 8:
            self._out.write(bytes([self._buffer]))
            self._buffer = 0
            self._bits_in_buffer = 0

    def close(self) -> None:
        if self._closed:
            return
        # Pad remaining bits with zeros and flush
        if self._bits_in_buffer > 0:
            self._buffer <<= (8 - self._bits_in_buffer)
            self._out.write(bytes([self._buffer]))
        self._closed = True

    @property
    def stream(self) -> io.BytesIO:
        return self._out


class BitInputStream:
    """Reads individual bits from a byte stream."""

    def __init__(self, data: bytes) -> None:
        self._data = data
        self._byte_pos = 0
        self._bit_pos = 0  # 0-7, MSB first

    def read_bit(self) -> int:
        if self._byte_pos >= len(self._data):
            return 0  # EOF pads with zeros
        bit = (self._data[self._byte_pos] >> (7 - self._bit_pos)) & 1
        self._bit_pos += 1
        if self._bit_pos == 8:
            self._bit_pos = 0
            self._byte_pos += 1
        return bit

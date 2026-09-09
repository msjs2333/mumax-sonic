"""Bounded reader for one rectangular OOMMF OVF 2.0 vector field.

The reader deliberately stops at the file-format boundary: it preserves every
numeric vector value (including zero and NaN), and does not infer material,
entities, or time semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import numpy as np


MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_VECTOR_BYTES = 64 * 1024 * 1024
_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_TIME_DESC = re.compile(rf"Desc:\s+Total simulation time:\s+({_NUMBER})\s+s\s*\Z")
_UNITLESS_TIME_DESC = re.compile(rf"Desc:\s+Total simulation time:\s+({_NUMBER})\s*\Z")
_DATA_BEGIN = re.compile(r"Begin:\s+Data\s+(Text|Binary\s+[48])\s*\Z", re.I)


@dataclass(frozen=True)
class OVFData:
    """Raw OVF field and its SI grid metadata.

    ``origin_m`` is the first sample centre (the OVF ``*base`` values), and
    vectors use array axes ``(z, y, x, component)`` with XYZ components.
    """

    vectors: np.ndarray
    step_m: tuple[float, float, float]
    origin_m: tuple[float, float, float]
    time_s: float | None
    labels: tuple[str, str, str]
    units: tuple[str, str, str]
    sha256: str
    encoding: str
    time_hint: tuple[str, ...] = ()
    byte_count: int = 0


def read_ovf(path) -> OVFData:
    """Read a bounded, single-segment rectangular OVF 2.0 vector field."""
    source = Path(path)
    try:
        size = source.stat().st_size
    except OSError as exc:
        raise ValueError("cannot stat OVF file") from exc
    if size > MAX_FILE_BYTES:
        raise ValueError("OVF exceeds 64 MiB file limit")
    try:
        with source.open('rb') as stream:
            raw = stream.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise ValueError("cannot read OVF file") from exc
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("OVF exceeds 64 MiB file limit")

    digest = hashlib.sha256(raw).hexdigest()
    pos, first = _next_record(raw, 0)
    if _normal(_comment(first)) != "oommf ovf 2.0":
        raise ValueError("only OOMMF OVF 2.0 is supported")
    pos, segment_count = _next_record(raw, pos)
    if _normal(_comment(segment_count)) != "segment count: 1":
        raise ValueError("OVF must declare exactly one segment")
    pos, segment = _next_record(raw, pos)
    if _normal(_comment(segment)) != "begin: segment":
        raise ValueError("missing OVF segment")
    pos, header_start = _next_record(raw, pos)
    if _normal(_comment(header_start)) != "begin: header":
        raise ValueError("missing OVF header")

    header: dict[str, str] = {}
    times: list[float] = []
    unitless_times: list[float] = []
    time_hints: list[str] = []
    while True:
        pos, line = _next_record(raw, pos)
        raw_text = _comment_raw(line)
        text = raw_text.strip()
        if _normal(text) == "end: header":
            break
        text = text.split("##", 1)[0].rstrip()
        time_match = _TIME_DESC.fullmatch(text)
        if time_match:
            value = float(time_match.group(1))
            if not np.isfinite(value):
                raise ValueError("non-finite OVF simulation time")
            times.append(value)
            continue
        unitless_match = _UNITLESS_TIME_DESC.fullmatch(text)
        if unitless_match:
            value = float(unitless_match.group(1))
            if not np.isfinite(value):
                raise ValueError("non-finite OVF simulation time hint")
            unitless_times.append(value)
            time_hints.append(raw_text.lstrip())
            continue
        if text.startswith("Desc: Total simulation time:"):
            raise ValueError("malformed OVF simulation time")
        if ":" not in text:
            raise ValueError("malformed OVF header line")
        key, value = text.split(":", 1)
        key = key.strip().lower()
        value = value.strip()
        if not key or not value:
            raise ValueError("malformed OVF header line")
        if key in _CRITICAL_HEADERS:
            if key in header:
                raise ValueError(f"duplicate critical OVF header: {key}")
            header[key] = value

    if times and any(value != times[0] for value in times[1:]):
        raise ValueError("conflicting OVF simulation times")
    if unitless_times and any(value != unitless_times[0] for value in unitless_times[1:]):
        raise ValueError("conflicting unitless OVF simulation time hints")
    if times and unitless_times:
        raise ValueError("ambiguous mixed OVF simulation time units")
    time_s = times[0] if times else None
    geometry, labels, units = _validate_header(header)
    nx, ny, nz, step_m, origin_m = geometry
    values_count = nx * ny * nz * 3
    if values_count * np.dtype(np.float64).itemsize > MAX_VECTOR_BYTES:
        raise ValueError("decoded OVF vectors exceed 64 MiB limit")

    pos, data_start = _next_record(raw, pos)
    match = _DATA_BEGIN.fullmatch(_comment(data_start))
    if not match:
        raise ValueError("missing or unsupported OVF data section")
    encoding = match.group(1).lower().replace(" ", "")
    if encoding == "text":
        values, pos = _read_text(raw, pos, values_count)
    elif encoding == "binary4":
        values, pos = _read_binary(raw, pos, values_count, 4, 1234567.0)
    else:
        values, pos = _read_binary(raw, pos, values_count, 8, 123456789012345.0)

    pos = _consume_data_end(raw, pos, encoding)
    pos, segment_end = _next_record(raw, pos)
    if _normal(_comment(segment_end)) != "end: segment":
        raise ValueError("missing OVF segment trailer")
    if not _only_ignorable(raw, pos):
        raise ValueError("extra data after OVF segment")
    vectors = values.reshape((nz, ny, nx, 3))
    vectors.setflags(write=False)
    return OVFData(vectors, step_m, origin_m, time_s, labels, units, digest, encoding,
                   tuple(time_hints), len(raw))


_CRITICAL_HEADERS = frozenset({
    "meshtype", "meshunit", "xbase", "ybase", "zbase", "xstepsize",
    "ystepsize", "zstepsize", "xnodes", "ynodes", "znodes", "valuedim",
    "valuelabels", "valueunits",
})


def _line(raw: bytes, pos: int) -> tuple[int, bytes]:
    end = raw.find(b"\n", pos)
    if end < 0:
        if pos < len(raw):
            return len(raw), raw[pos:].rstrip(b"\r")
        raise ValueError("truncated OVF line")
    return end + 1, raw[pos:end].rstrip(b"\r")


def _comment(line: bytes) -> str:
    return _comment_raw(line).strip()


def _comment_raw(line: bytes) -> str:
    try:
        text = line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("OVF header must be UTF-8") from exc
    if not text.lstrip().startswith("#"):
        raise ValueError("OVF structural lines must be comments")
    marker = text.index("#")
    return text[marker + 1:]


def _normal(text: str) -> str:
    return " ".join(text.split()).casefold()


def _next_record(raw: bytes, pos: int) -> tuple[int, bytes]:
    """Return the next structural/header record, skipping OVF blank comments."""
    while pos < len(raw):
        pos, line = _line(raw, pos)
        stripped = line.strip()
        if _ignorable_record(stripped):
            continue
        return pos, line
    raise ValueError("truncated OVF record")


def _only_ignorable(raw: bytes, pos: int) -> bool:
    while pos < len(raw):
        pos, line = _line(raw, pos)
        stripped = line.strip()
        if not _ignorable_record(stripped):
            return False
    return True


def _ignorable_record(stripped: bytes) -> bool:
    return not stripped or stripped == b"#" or stripped.startswith(b"##")


def _validate_header(header):
    missing = _CRITICAL_HEADERS.difference(header)
    if missing:
        raise ValueError("missing required OVF header")
    if header["meshtype"].lower() != "rectangular":
        raise ValueError("only rectangular OVF meshes are supported")
    if header["valuedim"] != "3":
        raise ValueError("OVF valuedim must be 3")
    unit_scale = {"m": 1.0, "nm": 1e-9, "um": 1e-6}.get(header["meshunit"].lower())
    if unit_scale is None:
        raise ValueError("OVF meshunit must be m, nm, or um")
    try:
        nodes = tuple(int(header[axis + "nodes"]) for axis in "xyz")
    except ValueError as exc:
        raise ValueError("OVF node counts must be integers") from exc
    if any(value <= 0 for value in nodes):
        raise ValueError("OVF node counts must be positive")
    try:
        bases = tuple(float(header[axis + "base"]) * unit_scale for axis in "xyz")
        steps = tuple(float(header[axis + "stepsize"]) * unit_scale for axis in "xyz")
    except ValueError as exc:
        raise ValueError("invalid OVF geometry") from exc
    if not all(np.isfinite(value) for value in bases) or not all(np.isfinite(value) and value > 0 for value in steps):
        raise ValueError("OVF bases must be finite and steps positive finite")
    labels = _tcl_words(header["valuelabels"])
    units = _tcl_words(header["valueunits"])
    if len(labels) != 3:
        raise ValueError("OVF must have exactly three value labels")
    if len(units) == 1:
        units *= 3
    if len(units) != 3:
        raise ValueError("OVF must have one or three value units")
    if not all(labels) or not all(units):
        raise ValueError("empty OVF labels or units")
    return (*nodes, steps, bases), tuple(labels), tuple(units)


def _tcl_words(value: str) -> list[str]:
    """Parse a small Tcl-list subset without evaluating substitutions or scripts."""
    words: list[str] = []
    index = 0
    length = len(value)
    while index < length:
        while index < length and value[index].isspace():
            index += 1
        if index == length:
            break
        if value[index] == "{":
            depth, start = 1, index + 1
            index += 1
            while index < length and depth:
                if value[index] == "{":
                    depth += 1
                elif value[index] == "}":
                    depth -= 1
                index += 1
            if depth:
                raise ValueError("unclosed brace in OVF Tcl list")
            word = value[start:index - 1]
            if index < length and not value[index].isspace():
                raise ValueError("invalid OVF Tcl list")
        elif value[index] == '"':
            index += 1
            pieces: list[str] = []
            while index < length and value[index] != '"':
                if value[index] == "\\":
                    if index + 1 == length:
                        raise ValueError("invalid escape in OVF Tcl list")
                    index += 1
                pieces.append(value[index])
                index += 1
            if index == length:
                raise ValueError("unclosed quote in OVF Tcl list")
            index += 1
            word = "".join(pieces)
            if index < length and not value[index].isspace():
                raise ValueError("invalid OVF Tcl list")
        else:
            start = index
            while index < length and not value[index].isspace():
                if value[index] in "{}\"[];$":
                    raise ValueError("unsupported syntax in OVF Tcl list")
                index += 1
            word = value[start:index]
        words.append(word)
    return words


def _read_text(raw: bytes, pos: int, count: int) -> tuple[np.ndarray, int]:
    tokens: list[bytes] = []
    while True:
        line_start = pos
        pos, line = _line(raw, pos)
        stripped = line.strip()
        if not stripped or stripped.startswith(b"##"):
            continue
        if line.lstrip().startswith(b"#"):
            marker = _normal(_comment(line))
            if marker == "end: data text":
                pos = line_start
                break
            if marker.startswith("end:"):
                raise ValueError("missing OVF text data trailer")
            # Comments in the text block are not numeric payload.
            continue
        tokens.extend(line.split())
        if len(tokens) > count:
            raise ValueError("OVF text data has extra values")
    if len(tokens) != count:
        raise ValueError("OVF text data is truncated")
    try:
        values = np.asarray([float(token) for token in tokens], dtype=np.float64)
    except ValueError as exc:
        raise ValueError("invalid OVF text value") from exc
    return values, pos


def _read_binary(raw: bytes, pos: int, count: int, width: int, check: float) -> tuple[np.ndarray, int]:
    total = (count + 1) * width
    if len(raw) - pos < total:
        raise ValueError("OVF binary data is truncated")
    dtype = np.dtype("<f4" if width == 4 else "<f8")
    body = raw[pos:pos + total]
    values = np.frombuffer(body, dtype=dtype)
    if values[0] != check:
        raise ValueError("OVF binary check value or byte order is invalid")
    return np.array(values[1:], dtype=np.float64, copy=True), pos + total


def _consume_data_end(raw: bytes, pos: int, encoding: str) -> int:
    # MuMax may put the trailer immediately after binary values; other writers
    # place it on the next line.  Both are complete OVF encodings.
    expected = f"End: Data {encoding.title() if encoding == 'text' else 'Binary ' + encoding[-1]}"
    pos, line = _next_record(raw, pos)
    if _normal(_comment(line)) != _normal(expected):
        raise ValueError("missing OVF data trailer")
    return pos

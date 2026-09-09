import hashlib
import struct

import numpy as np
import pytest

from mumax_sonic.sources.ovf import read_ovf


def _ovf(values, encoding="Text", *, meshunit="nm", bases=(1, 2, 3),
         steps=(2, 3, 5), labels="m_x m_y m_z", units="1", time="2.5e-12"):
    nx, ny, nz = 3, 2, 2
    header = "\n".join((
        "# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment", "# Begin: Header",
        "# meshtype: rectangular", f"# meshunit: {meshunit}",
        *(f"# {axis}base: {value}" for axis, value in zip("xyz", bases)),
        *(f"# {axis}stepsize: {value}" for axis, value in zip("xyz", steps)),
        *(f"# {axis}nodes: {value}" for axis, value in zip("xyz", (nx, ny, nz))),
        "# valuedim: 3", f"# valuelabels: {labels}", f"# valueunits: {units}",
        *([] if time is None else [f"# Desc: Total simulation time: {time} s"]),
        "# End: Header", f"# Begin: Data {encoding}",
    )).encode() + b"\n"
    values = np.asarray(values).reshape(-1)
    if encoding == "Text":
        body = (" ".join(str(float(v)) for v in values) + "\n# End: Data Text\n").encode()
    elif encoding == "Binary 4":
        body = struct.pack("<f", 1234567.0) + values.astype("<f4").tobytes() + b"# End: Data Binary 4\n"
    else:
        body = struct.pack("<d", 123456789012345.0) + values.astype("<f8").tobytes() + b"# End: Data Binary 8\n"
    return header + body + b"# End: Segment\n"


def test_ovf2_orders_non_square_multilayer_xyz_and_metadata(tmp_path):
    values = np.arange(36, dtype=float)
    path = tmp_path / "field.ovf"
    raw = _ovf(values, labels='{m x} "m y" m_z', units='A/m', meshunit="um")
    path.write_bytes(raw)

    data = read_ovf(path)
    assert data.vectors.shape == (2, 2, 3, 3)
    assert np.array_equal(data.vectors[1, 1, 2], (33, 34, 35))
    assert np.allclose(data.step_m, (2e-6, 3e-6, 5e-6), rtol=0, atol=1e-21)
    assert np.allclose(data.origin_m, (1e-6, 2e-6, 3e-6), rtol=0, atol=1e-21)
    assert data.labels == ("m x", "m y", "m_z")
    assert data.units == ("A/m",) * 3
    assert data.time_s == 2.5e-12
    assert data.sha256 == hashlib.sha256(raw).hexdigest()
    assert data.encoding == "text"


def test_text_binary4_binary8_are_equivalent_and_binary_trailer_can_be_immediate(tmp_path):
    values = np.linspace(-1, 1, 36)
    results = []
    for encoding in ("Text", "Binary 4", "Binary 8"):
        path = tmp_path / f"{encoding[-1]}.ovf"
        path.write_bytes(_ovf(values, encoding))
        results.append(read_ovf(path))
    assert np.allclose(results[0].vectors, results[1].vectors, rtol=0, atol=1e-7)
    assert np.allclose(results[0].vectors, results[2].vectors, rtol=0, atol=0)
    assert [item.encoding for item in results] == ["text", "binary4", "binary8"]


def test_nist_style_blank_and_double_hash_comments_and_lowercase_markers_are_valid(tmp_path):
    raw = _ovf(np.arange(36))
    raw = raw.replace(b"# OOMMF OVF 2.0\n", b"# OOMMF OVF 2.0\n#\n\n## descriptive comment\n")
    raw = raw.replace(b"# Segment count: 1", b"# segment count: 1")
    raw = raw.replace(b"# Begin: Segment", b"# begin: segment")
    raw = raw.replace(b"# Begin: Header", b"## header comment\n# begin: header")
    raw = raw.replace(b"# xnodes: 3", b"# xnodes: 3 ## numeric grid comment")
    raw = raw.replace(b"# End: Header", b"\n## before data\n# end: header")
    raw = raw.replace(b"# Begin: Data Text", b"# begin: data text\n## text payload comment")
    raw = raw.replace(b"# End: Data Text", b"## after text payload\n# end: data text")
    raw = raw.replace(b"# End: Segment\n", b"## segment comment\n# end: segment")
    path = tmp_path / "commented.ovf"
    path.write_bytes(raw)

    data = read_ovf(path)
    assert np.array_equal(data.vectors.ravel(), np.arange(36, dtype=float))
    assert data.encoding == "text"


@pytest.mark.parametrize("mutation, message", [
    (lambda raw: raw.replace(b"# valuedim: 3", b"# valuedim: 1"), "valuedim"),
    (lambda raw: raw.replace(b"# xnodes: 3", b"# xnodes: 3\n# xnodes: 3"), "duplicate"),
    (lambda raw: raw.replace(b"# End: Segment\n", b""), "truncated"),
    (lambda raw: raw + b"junk", "extra"),
    (lambda raw: raw.replace(b"# End: Data Text", b"# End: Data Binary 4"), "trailer"),
])
def test_malformed_or_partial_data_is_rejected(tmp_path, mutation, message):
    path = tmp_path / "bad.ovf"
    path.write_bytes(mutation(_ovf(np.arange(36))))
    with pytest.raises(ValueError, match=message):
        read_ovf(path)


def test_time_conflict_and_binary_bad_check_are_rejected(tmp_path):
    path = tmp_path / "conflict.ovf"
    raw = _ovf(np.arange(36)).replace(
        b"# Desc: Total simulation time: 2.5e-12 s",
        b"# Desc: Total simulation time: 2.5e-12 s\n# Desc: Total simulation time: 3e-12 s",
    )
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="conflicting"):
        read_ovf(path)
    path.write_bytes(_ovf(np.arange(36), "Binary 4").replace(struct.pack("<f", 1234567.0), b"\x00\x00\x00\x00", 1))
    with pytest.raises(ValueError, match="check"):
        read_ovf(path)


def test_duplicate_unitless_total_time_is_a_hint_not_seconds(tmp_path):
    path = tmp_path / "mumax_unitless_time.ovf"
    raw = _ovf(np.arange(36)).replace(
        b"# Desc: Total simulation time: 2.5e-12 s",
        b"# Desc: Total simulation time:  0  \n# Desc: Total simulation time:  0  ",
    )
    path.write_bytes(raw)

    data = read_ovf(path)
    assert data.time_s is None
    assert data.time_hint == (
        "Desc: Total simulation time:  0  ",
        "Desc: Total simulation time:  0  ",
    )

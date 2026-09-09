import struct

import numpy as np
import pytest

from mumax_sonic.sources.ovf import probe_ovf


def _binary(width=4, *, check=True):
    encoding = f"Binary {width}"
    lines = [
        "# OOMMF OVF 2.0", "# Segment count: 1", "# Begin: Segment", "# Begin: Header",
        "# meshtype: rectangular", "# meshunit: nm", "# xbase: 1", "# ybase: 2", "# zbase: 3",
        "# xstepsize: 4", "# ystepsize: 5", "# zstepsize: 6", "# xnodes: 2", "# ynodes: 1", "# znodes: 1",
        "# valuedim: 3", "# valuelabels: m_x m_y m_z", "# valueunits: 1",
        "# Desc: Total simulation time: 2e-12 s", "# End: Header", f"# Begin: Data {encoding}",
    ]
    value = 1234567.0 if width == 4 else 123456789012345.0
    pack = "<f" if width == 4 else "<d"
    dtype = "<f4" if width == 4 else "<f8"
    marker = struct.pack(pack, value if check else 0.0)
    return "\n".join(lines).encode() + b"\n" + marker + np.arange(6, dtype=dtype).tobytes() + (
        f"# End: Data {encoding}\n# End: Segment\n".encode()
    )


@pytest.mark.parametrize("width", [4, 8])
def test_binary_probe_checks_framing_and_returns_metadata_without_vectors(tmp_path, width):
    path = tmp_path / f"m000.{width}.ovf"
    raw = _binary(width)
    path.write_bytes(raw)
    probe = probe_ovf(path)
    assert probe.shape == (1, 1, 2, 3)
    assert probe.step_m == pytest.approx((4e-9, 5e-9, 6e-9))
    assert probe.origin_m == pytest.approx((1e-9, 2e-9, 3e-9))
    assert probe.time_s == pytest.approx(2e-12)
    assert probe.labels == ("m_x", "m_y", "m_z")
    assert probe.encoding == f"binary{width}"
    assert not hasattr(probe, "vectors")
    assert probe.byte_count == len(raw)


@pytest.mark.parametrize("width", [4, 8])
def test_binary_probe_rejects_bad_check_and_half_written_trailer(tmp_path, width):
    path = tmp_path / "bad.ovf"
    path.write_bytes(_binary(width, check=False))
    with pytest.raises(ValueError, match="check value"):
        probe_ovf(path)
    path.write_bytes(_binary(width)[:-18])
    with pytest.raises(ValueError, match="trailer|truncated"):
        probe_ovf(path)

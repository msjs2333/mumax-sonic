"""Invalid timing manifests should fail before starting a device or worker."""
import json
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('times,offsets,message', [
    ([0, 0], [0, 80], 'unique increasing physical times'),
    ([0, 1], [80, 0], 'nondecreasing wall offsets'),
    ([0, 1], [0, 1], 'source cadence must span'),
])
def test_invalid_continuous_input_reports_usage_error(tmp_path, times, offsets, message):
    path = tmp_path / 'bad.json'
    path.write_text(json.dumps(dict(frames=[dict(file='missing.ovf', time_s=t, wall_offset_s=w)
                                          for t, w in zip(times, offsets)])), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'check_ovf_continuous.py'
    result = subprocess.run([sys.executable, str(script), str(path), '--phase-s', '10',
                             '--report', str(tmp_path / 'report.json')],
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 2
    assert message in result.stderr
    assert 'Traceback' not in result.stderr

"""Strict diagnostic JSON: unavailable numerical results are null, never zero."""
import json
import math


def _finite_values(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_values(item) for item in value]
    return value


def report_json(payload):
    return json.dumps(_finite_values(payload), ensure_ascii=False, indent=2, allow_nan=False)

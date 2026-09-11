"""Measure activity kernel and view construction on a real XY crop."""
import argparse
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mumax_sonic.sources.ovf_replay import load_ovf_replay
from mumax_sonic.observers import activity
from mumax_sonic.observers.focused_activity import FocusedActivity, _crop
from mumax_sonic.field_pipeline import observe_field
from mumax_sonic.attention import Attention


def measure(call, repeats):
    times = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = call()
        times.append((time.perf_counter() - start) * 1000)
    return result, dict(samples_ms=times, median_ms=statistics.median(times))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest', type=Path)
    parser.add_argument('--baseline-kernel', type=Path)
    parser.add_argument('--baseline-pipeline', type=Path,
                        help='Optional earlier field_pipeline.py for same-kernel view comparison')
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error('repeats must be 1..20')
    previous, current = load_ovf_replay(args.manifest).frames[-2:]
    observer = FocusedActivity()
    observer.close()  # no concurrent background benchmark workload
    attention = Attention(radius=.4, extent_m=current.extent_m, origin_m=current.center_m[:2])
    bounds = observer._focus_bounds(current, attention)
    pair, crop_cost = measure(lambda: (_crop(previous, *bounds), _crop(current, *bounds)), args.repeats)
    before, after = pair
    report = dict(shape=after.vectors.shape, bounds=bounds, crop_pair=crop_cost,
                  note='Isolated archived real crop. No solver, UI or audio. Stages are separate repetitions; do not sum percentiles.')
    modules = [('current', activity)]
    if args.baseline_kernel:
        spec = importlib.util.spec_from_file_location('activity_baseline', args.baseline_kernel)
        baseline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline
        spec.loader.exec_module(baseline)
        modules.insert(0, ('baseline', baseline))
    reference = None
    for name, module in modules:
        kernel = module.angular_activity
        result, costs = measure(lambda: kernel(before, after), args.repeats)
        with patch.object(activity, 'angular_activity', kernel):
            view, view_cost = measure(lambda: observe_field(after, 'activity', previous=before), args.repeats)
            compact, compact_cost = measure(lambda: observe_field(
                after, 'activity', previous=before, activity_contributions=False), args.repeats)
        assert view.sample == compact.sample
        assert view.diagnostic == compact.diagnostic
        assert compact.contributions is None
        with patch.object(activity, 'angular_activity', return_value=result):
            _, full_build = measure(lambda: observe_field(after, 'activity', previous=before), args.repeats)
            _, compact_build = measure(lambda: observe_field(
                after, 'activity', previous=before, activity_contributions=False), args.repeats)
        report[name] = dict(kernel=costs, full_view=view_cost, coverage=result.coverage,
                            compact_view=compact_cost, full_build_only=full_build,
                            compact_build_only=compact_build,
                            mean_rad_s=result.mean_rad_s, max_rad_s=result.max_rad_s)
        if reference is None:
            reference = result
        else:
            np.testing.assert_array_equal(result.valid, reference.valid)
            np.testing.assert_allclose(result.rate_rad_s, reference.rate_rad_s,
                                       rtol=1e-9, atol=1e-3, equal_nan=True)
            assert result.validity == reference.validity
            assert result.warnings == reference.warnings
            report[name]['baseline_agreement'] = dict(
                rate_rtol=1e-9, rate_atol_rad_s=1e-3,
                max_absolute_difference_rad_s=float(np.nanmax(
                    np.abs(result.rate_rad_s - reference.rate_rad_s))))
        if hasattr(module, '_normalized'):
            _, normalization = measure(lambda: (module._normalized(before.vectors, before.mask),
                                               module._normalized(after.vectors, after.mask)), args.repeats)
            report[name]['normalization_pair'] = normalization
    if args.baseline_pipeline:
        spec = importlib.util.spec_from_file_location('mumax_sonic._baseline_pipeline', args.baseline_pipeline)
        baseline_pipeline = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = baseline_pipeline
        spec.loader.exec_module(baseline_pipeline)
        old, old_cost = measure(lambda: baseline_pipeline.observe_field(
            after, 'activity', previous=before), args.repeats)
        with patch.object(activity, 'angular_activity', return_value=result):
            _, old_build = measure(lambda: baseline_pipeline.observe_field(
                after, 'activity', previous=before), args.repeats)
        assert old.sample.validity == compact.sample.validity
        assert [o.source_id for o in old.sample.observations] == [o.source_id for o in compact.sample.observations]
        np.testing.assert_allclose([o.strength for o in old.sample.observations],
                                   [o.strength for o in compact.sample.observations], rtol=1e-12)
        np.testing.assert_allclose([o.position_m for o in old.sample.observations],
                                   [o.position_m for o in compact.sample.observations], rtol=1e-12, atol=1e-20)
        report['baseline_pipeline'] = dict(full_view=old_cost, build_only=old_build,
                                           compact_tiles_agree=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

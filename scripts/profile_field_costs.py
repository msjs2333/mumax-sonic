"""Read-only stage costs on an explicit two-or-more-frame OVF manifest."""
import argparse
from pathlib import Path
import sys
import time
import statistics
import json

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from mumax_sonic.sources.ovf_replay import load_ovf_replay
from mumax_sonic.field_pipeline import observe_field, apply_aggregation
from mumax_sonic.attention import Attention
from mumax_sonic.mapping import map_sample_with_report
from mumax_sonic.observers.focused_activity import FocusedActivity
from mumax_sonic.reporting import report_json
from check_live_soak import rss_bytes


def measured(call, count):
    samples=[]
    result=None
    for _ in range(count):
        start=time.perf_counter(); result=call()
        samples.append((time.perf_counter()-start)*1000)
    return result, dict(samples_ms=samples, median_ms=statistics.median(samples))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--radii',type=float,nargs='+',default=[.1,.2,.4])
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if not 1 <= args.repeats <= 20:
        parser.error('repeats must be 1..20')
    start_rss=rss_bytes()
    replay,loading=measured(lambda:load_ovf_replay(args.manifest),args.repeats)
    if len(replay.frames)<2:
        parser.error('two physical frames required')
    previous,current=replay.frames[-2:]
    attention=Attention(extent_m=current.extent_m,origin_m=current.center_m[:2])
    _,label=measured(lambda:current.with_source_kind('live'),args.repeats)
    full,physics=measured(lambda:observe_field(current,'activity',previous=previous),args.repeats)
    if full.sample.validity!='valid':
        raise ValueError('full activity invalid: verify physical time and explicit material mask')
    aggregated,aggregation=measured(lambda:apply_aggregation(full,attention,8,'adaptive'),args.repeats)
    _,mapping=measured(lambda:map_sample_with_report(aggregated.sample,attention,budget=8),args.repeats)
    focuses=[]
    for radius in args.radii:
        target=Attention(radius=radius,extent_m=current.extent_m,origin_m=current.center_m[:2])
        observer=FocusedActivity(background_interval_s=60)
        try:
            first,cold=measured(lambda:observer.observe(previous,current,target),1)
            deadline=time.monotonic()+15
            while time.monotonic()<deadline:
                view=observer.observe(previous,current,target)
                if view.diagnostic.get('focus_compute',{}).get('background_ready'):
                    break
                time.sleep(.02)
            else:
                raise RuntimeError('background did not finish')
            view,foreground=measured(lambda:observer.observe(previous,current,target),args.repeats)
            grouped,grouping=measured(lambda:apply_aggregation(view,target,8,'adaptive'),args.repeats)
            _,focused_mapping=measured(lambda:map_sample_with_report(grouped.sample,target,budget=8),args.repeats)
            focuses.append(dict(radius=radius,first_focus=cold,foreground=foreground,aggregation=grouping,
                mapping=focused_mapping,diagnostic=view.diagnostic['focus_compute'],
                compact_sites=view.contributions.positive.size,validity=view.sample.validity,
                mixed_sum=sum(o.strength for o in view.sample.observations),
                full_mean=full.diagnostic['mean_rad_s']))
        finally:
            observer.close()
    result=dict(input_shape=current.vectors.shape,sim_times_s=[previous.sim_time_s,current.sim_time_s],
        frame_bytes=current.vectors.nbytes+current.mask.nbytes,load_replay=loading,source_label=label,
        full_activity=physics,full_aggregation=aggregation,mapping=mapping,focus=focuses,
        rss_start_bytes=start_rss,rss_end_bytes=rss_bytes(),
        note='Existing physical frames; no GPU, UI or audio. Independent stage repetitions are not an end-to-end latency distribution. Warm foreground follows one background refresh; first-focus cost includes concurrent background competition.')
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(report_json(result),encoding='utf-8')
    print(report_json(result))


if __name__=='__main__':
    main()

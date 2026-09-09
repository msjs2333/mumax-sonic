"""Keep physical, spatial aggregation and voice-budget accounting separate."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest

from mumax_sonic.attention import Attention
from mumax_sonic.fields import FieldFrame
from mumax_sonic.field_pipeline import observe_field, field_selection_report
from mumax_sonic.mapping import map_sample_with_report
from mumax_sonic.sources.analytic import make_field
from mumax_sonic.sources.replay import save_replay
from mumax_sonic.ui.coverage import selection_summary


def rotation_pair():
    first=np.zeros((8,8,3));first[...,0]=1
    second=np.zeros_like(first);second[...,0]=np.cos(.1);second[...,1]=np.sin(.1)
    return FieldFrame(first,1e-9,1e-9,0),FieldFrame(second,1e-9,1e-9,1e-11,sequence=1)


def test_uniform_activity_budget_fraction_and_headroom():
    a,b=rotation_pair();view=observe_field(b,'activity',previous=a)
    reports=[]
    for budget in (1,2,4,8,16):
        mapped=map_sample_with_report(view.sample,Attention(background=1),budget=budget,strength_reference=1e10)
        report=field_selection_report(view,mapped.report)
        reports.append(report)
        assert report['selected_fraction_observer_input']['absolute']==pytest.approx(budget/16)
        assert sum(s.gain for s in mapped.scene.sources)<=.15
        assert view.diagnostic['mean_rad_s']==pytest.approx(1e10)
        totals=report['groups'][0]['absolute']
        assert totals['total']==pytest.approx(totals['selected']+totals['omitted_budget'])
    assert reports[-1]['groups'][0]['absolute']['omitted_budget']==0


def test_direction_circular_cancellation_is_spatial_loss_not_full_coverage():
    values=np.zeros((8,8,3));values[...,1]=np.where(np.indices((8,8)).sum(axis=0)%2,1,-1)
    view=observe_field(FieldFrame(values,1e-9,1e-9),'direction')
    assert view.sample.validity=='valid' and not view.sample.observations
    mapped=map_sample_with_report(view.sample,Attention(),budget=16)
    report=field_selection_report(view,mapped.report)
    assert report['spatial_aggregation']['channels']['absolute']==dict(input=1,represented=0,omitted=1)
    assert report['selected_fraction_observer_input']['absolute']==0
    assert '聚合退化省略' in selection_summary(report)


def test_topology_net_zero_does_not_make_fraction_undefined():
    values=make_field('opposite_pair')
    view=observe_field(FieldFrame(values,2e-6/64,2e-6/64,origin_m=(-1e-6,-1e-6,0)))
    attention=Attention(extent_m=1e-6,background=1)
    mapped=map_sample_with_report(view.sample,attention,budget=2)
    report=field_selection_report(view,mapped.report)
    assert abs(view.diagnostic['q_net'])<1e-12
    assert view.diagnostic['q_abs']==pytest.approx(2)
    assert report['selected_fraction_observer_input']['absolute']>0
    for channel in ('positive','negative'):
        assert report['spatial_aggregation']['channels'][channel]['input']==pytest.approx(1)
    assert 'Q−' in selection_summary(report)


def test_inspection_cli_budget_does_not_change_physics(tmp_path):
    replay=tmp_path/'rotation.npz';save_replay(replay,rotation_pair())
    root=Path(__file__).resolve().parents[1]
    records=[]
    for budget in (4,16):
        result=subprocess.run([sys.executable,str(root/'launch.py'),'--inspect-field',str(replay),
            '--recipe','activity','--source-budget',str(budget)],capture_output=True,text=True,encoding='utf-8',cwd=root)
        assert result.returncode==0,result.stderr
        frames=json.loads(result.stdout)['frames']
        assert frames[0]['selection']['validity']=='warming_up'
        records.append(frames[1])
    assert records[0]['mean_rad_s']==records[1]['mean_rad_s']
    assert len(records[0]['selection']['selected_ids'])==4
    assert len(records[1]['selection']['selected_ids'])==16

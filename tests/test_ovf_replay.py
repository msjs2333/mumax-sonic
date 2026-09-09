"""Independent OVF bytes test physical metadata and shared replay behaviour."""
import hashlib
import json
import struct
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mumax_sonic.sources.ovf_replay import load_ovf_replay, load_field_replay
from mumax_sonic.sources.replay import save_replay
from mumax_sonic.field_pipeline import observe_field
from mumax_sonic.mapping import map_sample
from mumax_sonic.attention import Attention


def test_small_input_read_does_not_request_the_entire_memory_budget(tmp_path, monkeypatch):
    from mumax_sonic.sources.ovf_replay import _bounded_bytes
    path = tmp_path / 'small.bin'
    payload = b'x' * 1024
    path.write_bytes(payload)
    original_open = Path.open
    requests = []
    class TrackedFile:
        def __init__(self, stream): self.stream = stream
        def __enter__(self): return self
        def __exit__(self, *args): self.stream.close()
        def fileno(self): return self.stream.fileno()
        def read(self, size=-1):
            requests.append(size)
            return self.stream.read(size)
    monkeypatch.setattr(Path, 'open', lambda self, *a, **kw: TrackedFile(original_open(self, *a, **kw)))
    assert _bounded_bytes(path, 256 * 1024 * 1024) == payload
    assert requests and all(0 <= size <= len(payload) for size in requests)


def write_ovf(path, values, time_s, *, labels='m_x m_y m_z', unit='1'):
    nz, ny, nx, _ = values.shape
    lines = ['# OOMMF OVF 2.0', '# Segment count: 1', '# Begin: Segment', '# Begin: Header',
        '# meshtype: rectangular', '# meshunit: nm', '# valuedim: 3',
        f'# valuelabels: {labels}', f'# valueunits: {unit}',
        f'# xnodes: {nx}', f'# ynodes: {ny}', f'# znodes: {nz}',
        '# xbase: -7', '# ybase: 11', '# zbase: 13',
        '# xstepsize: 2', '# ystepsize: 3', '# zstepsize: 5']
    if time_s is not None:
        lines.append(f'# Desc: Total simulation time: {time_s:.17g} s')
    raw = ('\n'.join(lines+['# End: Header', '# Begin: Data Binary 8', '']).encode()
           +struct.pack('<d',123456789012345)+np.asarray(values,dtype='<f8').tobytes()
           +b'# End: Data Binary 8\n# End: Segment\n')
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def manifest(tmp_path, *, count=3, layers=1, indices=None, time=True):
    records=[]
    for i in (range(count) if indices is None else indices):
        values=np.zeros((layers,3,4,3))
        values[...,0]=np.cos(i*.1)
        values[...,1]=np.sin(i*.1)
        if layers>1:
            values[1,...]=(0,0,1)
        path=tmp_path/f'm{i:06d}.ovf'
        digest=write_ovf(path,values,i*1e-11 if time else None)
        records.append(dict(file=path.name,sha256=digest,sequence=i))
    meta=dict(schema_version=1,entity_id='m',segment_id='test',time_kind='dynamics',origin='synthetic',
        quantity='magnetization_direction',value_unit='1',components=['x','y','z'],mask='all',frames=records)
    path=tmp_path/'sequence.json'
    path.write_text(json.dumps(meta),encoding='utf-8')
    return path,meta


def publish(path,meta):
    path.write_text(json.dumps(meta),encoding='utf-8')
    return load_ovf_replay(path)


def test_xyz_sample_centers_and_explicit_layer(tmp_path):
    path,meta=manifest(tmp_path,layers=2)
    with pytest.raises(ValueError,match='z_index'):
        load_field_replay(path)
    meta['z_index']=1
    replay=publish(path,meta)
    frame=replay.frames[0]
    assert frame.vectors.shape==(3,4,3)
    np.testing.assert_array_equal(frame.vectors,np.broadcast_to((0,0,1),(3,4,3)))
    assert frame.origin_m==pytest.approx((-7e-9,11e-9,18e-9))
    assert (frame.dx_m,frame.dy_m)==pytest.approx((2e-9,3e-9))
    assert json.loads(frame.provenance)['z_index']==1


def test_header_time_gap_and_npz_lineage(tmp_path):
    path,meta=manifest(tmp_path,indices=(0,1,4,5))
    replay=load_field_replay(path)
    a,b,c,d=replay.frames
    view=observe_field(b,'activity',previous=a)
    assert view.diagnostic['mean_rad_s']==pytest.approx(1e10)
    assert view.diagnostic['input']['ovf_sha256']==meta['frames'][1]['sha256']
    gap=observe_field(c,'activity',previous=b)
    assert gap.sample.validity=='warming_up' and not map_sample(gap.sample,Attention()).sources
    assert observe_field(d,'activity',previous=c).sample.validity=='valid'
    output=tmp_path/'cache.npz'
    save_replay(output,replay.frames)
    loaded=load_field_replay(output)
    assert observe_field(loaded.frames[1],'activity',previous=loaded.frames[0]).diagnostic['input']==view.diagnostic['input']


def test_missing_time_never_comes_from_filename(tmp_path):
    path,meta=manifest(tmp_path,time=False)
    with pytest.raises(ValueError,match='missing physical time'):
        load_ovf_replay(path)
    for i,record in enumerate(meta['frames']):
        record['time_s']=i*1e-11
    assert publish(path,meta).frames[-1].sim_time_s==pytest.approx(2e-11)


@pytest.mark.parametrize('change,match',[
    (lambda m:m['frames'][1].update(time_s=9e-9),'conflicts'),
    (lambda m:m['frames'][1].update(sha256='0'*64),'SHA256'),
    (lambda m:m.update(value_unit='A/m'),'units'),
    (lambda m:m.update(components=['z','y','x']),'labels'),
    (lambda m:m.pop('mask'),'mask'),
    (lambda m:m['frames'][1].update(sequence=0),'strictly increasing'),
    (lambda m:m.update(time_kind='unknown'),'time_kind'),
])
def test_manifest_disagreements_fail_closed(tmp_path,change,match):
    path,meta=manifest(tmp_path)
    change(meta)
    with pytest.raises(ValueError,match=match):
        publish(path,meta)


def test_zero_vector_is_invalid_until_explicit_material_mask(tmp_path):
    path,meta=manifest(tmp_path,count=1)
    values=np.zeros((1,3,4,3));values[...,0]=1;values[0,0,0]=0
    meta['frames'][0]['sha256']=write_ovf(tmp_path/meta['frames'][0]['file'],values,0)
    assert observe_field(publish(path,meta).frames[0]).sample.coverage<1
    mask=np.ones((3,4),dtype=bool);mask[0,0]=False
    maskpath=tmp_path/'mask.npy';np.save(maskpath,mask)
    meta['mask']=dict(file='mask.npy',sha256=hashlib.sha256(maskpath.read_bytes()).hexdigest())
    replay=publish(path,meta)
    assert not replay.frames[0].mask[0,0]
    assert json.loads(replay.frames[0].provenance)['mask_source']==meta['mask']['sha256']


def test_manifest_builder_and_inspection_cli(tmp_path):
    _,meta=manifest(tmp_path)
    output=tmp_path/'built.json'
    root=Path(__file__).resolve().parents[1]
    command=[sys.executable,str(root/'scripts/make_ovf_manifest.py'),str(tmp_path),str(output),
             '--origin','synthetic','--time-kind','dynamics','--all-material']
    result=subprocess.run(command,capture_output=True,text=True,cwd=root)
    assert result.returncode==0,result.stderr
    result=subprocess.run([sys.executable,str(root/'launch.py'),'--inspect-field',str(output),'--recipe','activity'],
                          capture_output=True,text=True,encoding='utf-8',cwd=root)
    assert result.returncode==0,result.stderr
    report=json.loads(result.stdout)
    assert report['frames'][1]['mean_rad_s']==pytest.approx(1e10)
    before=output.read_bytes()
    assert subprocess.run(command,capture_output=True,cwd=root).returncode!=0
    assert output.read_bytes()==before


def test_unitless_header_needs_explicit_time_and_hash_bound_evidence(tmp_path):
    path,meta=manifest(tmp_path,count=2)
    for record in meta['frames']:
        source=tmp_path/record['file']
        raw=source.read_bytes()
        start=raw.index(b'# Desc:')
        end=raw.index(b'\n',start)
        raw=raw[:start]+b'# Desc: Total simulation time: 0'+raw[end:]
        source.write_bytes(raw)
        record['sha256']=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError,match='missing physical time'):
        publish(path,meta)
    evidence=tmp_path/'source-times.json'
    evidence.write_text('[0, 1e-11]')
    meta['time_evidence']=dict(file=evidence.name,sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),description='Recorded solver times in seconds')
    for i,record in enumerate(meta['frames']):
        record['time_s']=i*1e-11
    replay=publish(path,meta)
    info=json.loads(replay.frames[1].provenance)
    assert info['header_time_s'] is None and info['header_time_hint']
    assert info['time_evidence_sha256']==meta['time_evidence']['sha256']
    evidence.write_text('[0, 1]')
    with pytest.raises(ValueError,match='time evidence SHA256'):
        load_ovf_replay(path)


def test_oversized_npy_header_is_rejected_before_allocation(tmp_path):
    import io
    path,meta=manifest(tmp_path,count=1)
    output=io.BytesIO()
    np.lib.format.write_array_header_1_0(output,dict(descr='|b1',fortran_order=False,shape=(10**12,10**12)))
    raw=output.getvalue()
    (tmp_path/'mask.npy').write_bytes(raw)
    meta['mask']=dict(file='mask.npy',sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ValueError,match='payload size'):
        publish(path,meta)


def test_duplicate_manifest_keys_are_ambiguous(tmp_path):
    path,_=manifest(tmp_path)
    raw=path.read_text().replace('"schema_version": 1','"schema_version": 2, "schema_version": 1')
    path.write_text(raw)
    with pytest.raises(ValueError,match='duplicate manifest key'):
        load_ovf_replay(path)

"""SDK half-split RoPE against the pinned official Qwen implementation."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare():
    import numpy as np
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding, apply_rotary_pos_emb
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'evidence' / ('qwen-position-' + stamp)
    out.mkdir()
    model = ROOT / 'models/qwen2.5-0.5b-7ae5576'
    trace = ROOT / 'evidence/reference-f32-greedy-20260908T0554/arithmetic/step-000.npz'
    ref = np.load(trace)
    q = ref['model.layers.0.self_attn.q_proj'][0, -1].copy()
    k = ref['model.layers.0.self_attn.k_proj'][0, -1].copy()
    rotary = Qwen2RotaryEmbedding(AutoConfig.from_pretrained(model, local_files_only=True))
    frequency = rotary.inv_freq.numpy().copy()
    positions = [0, 1, 29, 30, 31, 32, 1023, 2047, 29]
    expected_q, expected_k = [], []
    for position in positions:
        qt = torch.from_numpy(q.reshape(1, 14, 1, 64))
        kt = torch.from_numpy(k.reshape(1, 2, 1, 64))
        cosine, sine = rotary(qt, torch.tensor([[position]]))
        qo, ko = apply_rotary_pos_emb(qt, kt, cosine, sine)
        expected_q.append(qo.numpy().ravel())
        expected_k.append(ko.numpy().ravel())
    np.savez(out/'inputs.npz', query=q, key=k, frequency=frequency,
             expected_q=np.stack(expected_q), expected_k=np.stack(expected_k))
    for source, name in [(Path(__file__), 'driver.py'), (ROOT/'tools/sdk_probe.py','executor.py'),
                         (ROOT/'csl/kernels/qwen_position_f32.csl','kernel.csl'),
                         (ROOT/'configs/precision.json','precision.json')]:
        shutil.copy2(source, out/name)
    (out/'config.json').write_text(json.dumps(dict(positions=positions, in_place=[False]*8+[True],
        model_config_sha256=digest(model/'config.json'), trace_sha256=digest(trace),
        scope='Full14Q/2K heads, head64, official FP32 rotary equations at boundary positions; final case aliases Q and K outputs separately. Not KV or attention qualification.'),indent=2)+'\n')
    (out/'pe.csl').write_text('''param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const clock=@import_module("<time>");
const qr=@import_module("kernel.csl",.{.heads=14});const kr=@import_module("kernel.csl",.{.heads=2});
var query=@zeros([896]f32);var key=@zeros([128]f32);var frequency=@zeros([32]f32);var query_out=@zeros([896]f32);var key_out=@zeros([128]f32);
var control=@zeros([2]u32);var progress=@zeros([1]u32);var timing=@zeros([6]u16);var started=@zeros([3]u16);var finished=@zeros([3]u16);
fn compute() void {
 clock.enable_tsc();clock.get_timestamp(&started);
 qr.prepare(@as(u16,control[0]),&frequency);kr.prepare(@as(u16,control[0]),&frequency);
 if(control[1]==0){qr.apply(&query,&query_out);kr.apply(&key,&key_out);}
 else{qr.apply(&query,&query);kr.apply(&key,&key);const q=@get_dsd(mem1d_dsd,.{.base_address=&query,.extent=896});const qo=@get_dsd(mem1d_dsd,.{.base_address=&query_out,.extent=896});const k=@get_dsd(mem1d_dsd,.{.base_address=&key,.extent=128});const ko=@get_dsd(mem1d_dsd,.{.base_address=&key_out,.extent=128});@mov32(qo,q);@mov32(ko,k);}
 clock.get_timestamp(&finished);clock.disable_tsc();for(@range(u16,3))|i|{timing[i]=started[i];timing[i+3]=finished[i];}progress[0]+=1;sys.unblock_cmd_stream();
}
const qp:[*]f32=&query;const kp:[*]f32=&key;const fp:[*]f32=&frequency;const qo:[*]f32=&query_out;const ko:[*]f32=&key_out;const cp:[*]u32=&control;const pp:[*]u32=&progress;const tp:[*]u16=&timing;
comptime{@export_symbol(qp,"query");@export_symbol(kp,"key");@export_symbol(fp,"frequency");@export_symbol(qo,"query_out");@export_symbol(ko,"key_out");@export_symbol(cp,"control");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(compute);}
''')
    exports = ''.join(f'@export_name("{name}",[*]{kind},false);' for name, kind in
        [('query','f32'),('key','f32'),('frequency','f32'),('query_out','f32'),('key_out','f32'),('control','u32'),('progress','u32'),('timing','u16')])
    (out/'layout.csl').write_text('const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});'+exports+'@export_name("compute",fn()void);}\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    verify(root)
    os.chdir(root)
    command=['cslc','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph']
    (root/'compile-command.json').write_text(json.dumps(command)+'\n')
    subprocess.run(command,check=True)
    c=json.loads((root/'config.json').read_text());data=np.load(root/'inputs.npz')
    gate=json.loads((root/'precision.json').read_text())['stage_gate']['absolute_or_normwise']
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3))
    ids={n:runner.get_id(n) for n in ['query','key','frequency','query_out','key_out','control','progress','timing']}
    runner.load();runner.run();rows=[]
    try:
        for epoch,(position,alias) in enumerate(zip(c['positions'],c['in_place'])):
            for name,value in [('query',data['query']),('key',data['key']),('frequency',data['frequency']),('control',np.array([position,int(alias)],np.uint32))]:
                runner.memcpy_h2d(ids[name],value.copy(),0,0,1,1,len(value),streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            runner.launch('compute',nonblock=False);actual={};checks={}
            for name,length in [('query_out',896),('key_out',128),('progress',1),('timing',6)]:
                value=np.zeros(length,np.float32 if name.endswith('_out') else np.uint32)
                runner.memcpy_d2h(value,ids[name],0,0,1,1,length,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT if name=='timing' else MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
                actual[name]=value
            np.savez(root/f'actual-{epoch}.npz',**actual)
            for name,key in [('query_out','expected_q'),('key_out','expected_k')]:
                got=actual[name].astype(np.float64);expected=data[key][epoch].astype(np.float64);assert np.isfinite(got).all()
                absolute=float(np.max(np.abs(got-expected)));l2=float(np.linalg.norm(got-expected)/max(np.linalg.norm(expected),1e-30));peak=float(absolute/max(np.max(np.abs(expected)),1e-30))
                assert absolute<=gate['max_abs'] or (l2<=gate['both']['relative_l2'] and peak<=gate['both']['relative_peak']),(name,position,absolute,l2,peak)
                checks[name]=dict(max_abs=absolute,relative_l2=l2,relative_peak=peak)
            assert actual['progress'][0]==epoch+1
            t=actual['timing'];assert np.all(t<65536)
            ticks=lambda a:int(a[0])+(int(a[1])<<16)+(int(a[2])<<32)
            cycles=(ticks(t[3:])-ticks(t[:3]))&((1<<48)-1);assert 0<cycles<1<<40
            rows.append(dict(position=position,in_place=alias,checks=checks,cycles=cycles,actual_sha256=digest(root/f'actual-{epoch}.npz')))
            (root/'results.json').write_text(json.dumps(dict(success=False,cases=rows),indent=2)+'\n')
            print('POSITION',position,'in_place',alias,flush=True)
    finally:
        runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=rows,scope=c['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()))
        from executor import execute
        execute(a.execute.resolve(),1800)

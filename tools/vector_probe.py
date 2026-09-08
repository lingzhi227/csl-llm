"""Real Qwen vector-stage fixtures and finite adversarial SDK cases."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def prepare():
    import numpy as np
    from safetensors.torch import load_file
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=ROOT/'evidence'/('vector-f32-'+stamp);out.mkdir()
    model=ROOT/'models/qwen2.5-0.5b-7ae5576/model.safetensors'
    trace=ROOT/'evidence/reference-f32/arithmetic/step-000.npz'
    weights=load_file(model);ref=np.load(trace)
    cases=[];inputs=[];others=[];gammas=[];expected=[]
    def add(name,op,x,z=None,g=None,official=None):
        x=np.asarray(x,np.float32);z=np.zeros(896,np.float32) if z is None else np.asarray(z,np.float32);g=np.ones(896,np.float32) if g is None else np.asarray(g,np.float32)
        assert x.shape==z.shape==g.shape==(896,)
        a=x.astype(np.float64);b=z.astype(np.float64)
        if op in (0,4): y=a/np.sqrt(np.mean(a*a)+1e-6)*g.astype(np.float64)
        elif op==1:
            e=np.exp(-np.abs(a));y=np.where(a>=0,a,a*e)/(1+e)*b
        elif op==2:
            e=np.exp(a-np.max(a));y=e/np.sum(e)
        else:y=a+b
        cases.append(dict(name=name,operation=op,official_reference=official is not None))
        inputs.append(x);others.append(z);gammas.append(g);expected.append(y if official is None else np.asarray(official,np.float64))
    for layer in (0,12,23):
        prefix=f'model.layers.{layer}'
        x=ref['model.embed_tokens' if layer==0 else f'model.layers.{layer-1}'][0,-1]
        g=weights[prefix+'.input_layernorm.weight'].float().numpy()
        add(f'layer{layer}-input-rms',0,x,g=g,official=ref[prefix+'.input_layernorm'][0,-1])
        residual=(x+ref[prefix+'.self_attn.o_proj'][0,-1]).astype(np.float32)
        add(f'layer{layer}-residual',3,x,ref[prefix+'.self_attn.o_proj'][0,-1],official=residual)
        g=weights[prefix+'.post_attention_layernorm.weight'].float().numpy()
        add(f'layer{layer}-post-attention-rms',0,residual,g=g,official=ref[prefix+'.post_attention_layernorm'][0,-1])
        add(f'layer{layer}-swiglu-slice',1,ref[prefix+'.mlp.gate_proj'][0,-1,:896],ref[prefix+'.mlp.up_proj'][0,-1,:896])
    add('rms-zero',0,np.zeros(896));add('rms-tiny',0,np.linspace(-1e-6,1e-6,896));add('rms-large',0,np.linspace(-2048,2048,896))
    add('swiglu-extremes',1,np.linspace(-100,100,896),np.linspace(-2,2,896))
    add('softmax-constant',2,np.full(896,13));add('softmax-wide',2,np.linspace(-100,100,896));add('softmax-two-equal-maxima',2,np.where(np.isin(np.arange(896),[0,895]),30.,-30.))
    add('repeat-first-rms',0,inputs[0].copy(),g=gammas[0].copy(),official=expected[0].copy())
    add('rms-in-place',4,inputs[0].copy(),g=gammas[0].copy(),official=expected[0].copy())
    np.savez(out/'inputs.npz',inputs=np.stack(inputs),others=np.stack(others),gammas=np.stack(gammas),expected=np.stack(expected))
    for source,name in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/kernels/vector_f32.csl','kernel.csl'),(ROOT/'configs/precision.json','precision.json')]:shutil.copy2(source,out/name)
    (out/'pe.csl').write_text('''param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const clock=@import_module("<time>");const k=@import_module("kernel.csl",.{.length=896});
var input=@zeros([896]f32);var other=@zeros([896]f32);var gamma=@zeros([896]f32);var output=@zeros([896]f32);
var mode=@zeros([1]u32);var progress=@zeros([1]u32);var timing=@zeros([6]u16);var started=@zeros([3]u16);var finished=@zeros([3]u16);
fn compute() void {
 clock.enable_tsc();clock.get_timestamp(&started);
 if(mode[0]==0){k.rmsnorm(&input,&gamma,&output);}else if(mode[0]==1){k.swiglu(&input,&other,&output);}else if(mode[0]==2){k.softmax(&input,&output);}else if(mode[0]==4){k.rmsnorm(&input,&gamma,&input);const src=@get_dsd(mem1d_dsd,.{.base_address=&input,.extent=896});const dst=@get_dsd(mem1d_dsd,.{.base_address=&output,.extent=896});@mov32(dst,src);}else{k.residual(&input,&other,&output);}
 clock.get_timestamp(&finished);clock.disable_tsc();for(@range(u16,3))|i|{timing[i]=started[i];timing[i+3]=finished[i];}progress[0]+=1;sys.unblock_cmd_stream();
}
const xp:[*]f32=&input;const zp:[*]f32=&other;const gp:[*]f32=&gamma;const yp:[*]f32=&output;const mp:[*]u32=&mode;const pp:[*]u32=&progress;const tp:[*]u16=&timing;
comptime{@export_symbol(xp,"input");@export_symbol(zp,"other");@export_symbol(gp,"gamma");@export_symbol(yp,"output");@export_symbol(mp,"mode");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(compute);}
''')
    (out/'layout.csl').write_text('''const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("input",[*]f32,false);@export_name("other",[*]f32,false);@export_name("gamma",[*]f32,false);@export_name("output",[*]f32,false);@export_name("mode",[*]u32,false);@export_name("progress",[*]u32,false);@export_name("timing",[*]u16,false);@export_name("compute",fn()void);}
''')
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    (out/'config.json').write_text(json.dumps(dict(cases=cases,model_sha256=sha(model),trace_sha256=sha(trace),length=896,softmax_structure=dict(nonnegative=True,sum_absolute_tolerance=2e-5),scope='Three native-hidden-width layer RMS/residual checks; partial FFN-width SwiGLU and dense finite softmax adversarial checks. Not a full layer, attention or model.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify,sha
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root)
    cmd=['cslc','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');subprocess.run(cmd,check=True)
    c=json.loads((root/'config.json').read_text());gate=json.loads((root/'precision.json').read_text())['stage_gate']['absolute_or_normwise'];data=np.load(root/'inputs.npz');runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['input','other','gamma','output','mode','progress','timing']};runner.load();runner.run();rows=[]
    try:
        for epoch,case in enumerate(c['cases']):
            for name,value in [('input',data['inputs'][epoch]),('other',data['others'][epoch]),('gamma',data['gammas'][epoch]),('mode',np.array([case['operation']],np.uint32))]:
                runner.memcpy_h2d(ids[name],value.copy(),0,0,1,1,len(value),streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            runner.launch('compute',nonblock=False);actual={}
            for name,n in [('output',896),('progress',1),('timing',6)]:
                value=np.zeros(n,np.float32 if name=='output' else np.uint32);runner.memcpy_d2h(value,ids[name],0,0,1,1,n,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT if name=='timing' else MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);actual[name]=value
            np.savez(root/f'actual-{epoch}.npz',**actual);got=actual['output'].astype(np.float64);expected=data['expected'][epoch];assert np.isfinite(got).all();error=got-expected
            absolute=float(np.max(np.abs(error)));l2=float(np.linalg.norm(error)/max(np.linalg.norm(expected),1e-30));peak=float(absolute/max(np.max(np.abs(expected)),1e-30));assert absolute<=gate['max_abs'] or (l2<=gate['both']['relative_l2'] and peak<=gate['both']['relative_peak']),(case['name'],absolute,l2,peak)
            if case['operation']==2:
                assert np.all(got>=0.0) and abs(float(np.sum(got))-1.0)<=2e-5,('probability mass',case['name'],float(np.sum(got)))
            assert actual['progress'][0]==epoch+1;t=actual['timing'].astype(np.uint64);assert np.all(t<65536);ticks=lambda a:int(a[0])+(int(a[1])<<16)+(int(a[2])<<32);cycles=(ticks(t[3:])-ticks(t[:3]))&((1<<48)-1);assert 0<cycles<1<<40
            rows.append(dict(name=case['name'],operation=case['operation'],max_abs=absolute,relative_l2=l2,relative_peak=peak,cycles=cycles,actual_sha256=sha(root/f'actual-{epoch}.npz')));(root/'results.json').write_text(json.dumps(dict(success=False,cases=rows),indent=2)+'\n');print(case['name'],absolute,l2,cycles,flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=rows,scope=c['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

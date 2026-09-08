"""Exact BF16 tied embedding lookup and deterministic device argmax/merge."""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    import numpy as np
    from safetensors.torch import load_file
    out=ROOT/'evidence'/('selection-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    model=ROOT/'models/qwen2.5-0.5b-7ae5576/model.safetensors';trace=ROOT/'evidence/reference-f32/arithmetic/step-000.npz';base=151552
    weights=load_file(model)['model.embed_tokens.weight'][base:base+128,:112].float().numpy();ref=np.load(trace);logits=ref['logits'].reshape(-1,151936)[-1,base:base+128].copy()
    vectors=[logits,logits,logits,np.full(128,-7,np.float32),np.zeros(128,np.float32),np.linspace(-10,10,128,dtype=np.float32),np.array([-0.]+[0.]*127,np.float32)]
    rows=[0,1,63,93,94,127,0];cases=[];candidate=[];expected_index=[];expected_value=[]
    for n,values in enumerate(vectors):
        index=int(np.argmax(values))+base;value=values[index-base]
        ci=index if n==0 else 0 if n==1 else 151935
        cv=value if n<=2 else np.float32(-1e10)
        merged=ci if cv>value or (cv==value and ci<index) else index
        cases.append(dict(name=['official-logits','lower-index-tie','higher-index-tie','negative-constant','all-zero','last-maximum','signed-zero-tie'][n],row=rows[n],candidate_index=ci))
        candidate.append(cv);expected_index.append(merged);expected_value.append(cv if merged==ci else value)
    np.savez(out/'inputs.npz',weights=weights,packed=(weights.T.copy().view(np.uint32)>>16).astype(np.uint16),values=np.stack(vectors),candidate=np.array(candidate,np.float32),expected_lookup=weights[rows],expected_index=np.array(expected_index,np.uint32),expected_value=np.array(expected_value,np.float32))
    for src,dst in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/kernels/tied_embedding_f32.csl','embedding.csl'),(ROOT/'csl/kernels/argmax_f32.csl','argmax.csl')]:shutil.copy2(src,out/dst)
    (out/'config.json').write_text(json.dumps(dict(base=base,cases=cases,model_sha256=digest(model),trace_sha256=digest(trace),scope='One actual128x112 tied embedding tile including EOS row; exact lookup and128-value FP32 argmax with global-index pair merge. Not full vocabulary reduction or model generation.'),indent=2)+'\n')
    (out/'pe.csl').write_text('''param memcpy_params;const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const e=@import_module("embedding.csl",.{});const a=@import_module("argmax.csl");
var weights=@zeros([14336]u16);var values=@zeros([128]f32);var candidate=@zeros([1]f32);var lookup=@zeros([112]f32);var best=@zeros([1]f32);var index=@zeros([1]u32);var control=@zeros([3]u32);var progress=@zeros([1]u32);
fn compute() void {e.lookup(&weights,@as(u16,control[0]),&lookup);a.select(&values,128,control[1],&best[0],&index[0]);a.merge(candidate[0],control[2],&best[0],&index[0]);progress[0]+=1;sys.unblock_cmd_stream();}
const wp:[*]u16=&weights;const vp:[*]f32=&values;const cp:[*]f32=&candidate;const lp:[*]f32=&lookup;const bp:[*]f32=&best;const ip:[*]u32=&index;const ctrl:[*]u32=&control;const pp:[*]u32=&progress;
comptime{@export_symbol(wp,"weights");@export_symbol(vp,"values");@export_symbol(cp,"candidate");@export_symbol(lp,"lookup");@export_symbol(bp,"best");@export_symbol(ip,"index");@export_symbol(ctrl,"control");@export_symbol(pp,"progress");@export_symbol(compute);}
''')
    names=[('weights','u16'),('values','f32'),('candidate','f32'),('lookup','f32'),('best','f32'),('index','u32'),('control','u32'),('progress','u32')]
    (out/'layout.csl').write_text('const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});'+''.join(f'@export_name("{n}",[*]{t},false);' for n,t in names)+'@export_name("compute",fn()void);}\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);c=json.loads((root/'config.json').read_text());data=np.load(root/'inputs.npz')
    cmd=['cslc','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');subprocess.run(cmd,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['weights','values','candidate','lookup','best','index','control','progress']};runner.load();runner.run();reports=[]
    try:
        runner.memcpy_h2d(ids['weights'],data['packed'].astype(np.uint32).ravel(),0,0,1,1,14336,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        for epoch,case in enumerate(c['cases']):
            for name,value in [('values',data['values'][epoch]),('candidate',data['candidate'][epoch:epoch+1]),('control',np.array([case['row'],c['base'],case['candidate_index']],np.uint32)),('lookup',np.full(112,np.nan,np.float32))]:
                runner.memcpy_h2d(ids[name],value.copy(),0,0,1,1,len(value),streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            runner.launch('compute',nonblock=False);actual={}
            for name,n in [('lookup',112),('best',1),('index',1),('progress',1)]:
                value=np.zeros(n,np.float32 if name in ('lookup','best') else np.uint32);runner.memcpy_d2h(value,ids[name],0,0,1,1,n,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);actual[name]=value
            np.savez(root/f'actual-{epoch}.npz',**actual)
            assert np.array_equal(actual['lookup'].view(np.uint32),data['expected_lookup'][epoch].view(np.uint32))
            assert actual['index'][0]==data['expected_index'][epoch] and actual['best'][0]==data['expected_value'][epoch]
            assert actual['progress'][0]==epoch+1
            reports.append(dict(name=case['name'],index=int(actual['index'][0]),lookup_exact=True,actual_sha256=digest(root/f'actual-{epoch}.npz')));print('SELECTION',case['name'],flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=reports,scope=c['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

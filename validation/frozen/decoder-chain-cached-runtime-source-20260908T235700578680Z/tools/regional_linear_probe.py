"""Two offset, independent resident contractions with device-side repeated epochs."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def prepare():
    import numpy as np
    from safetensors.torch import load_file
    out=ROOT/'evidence'/('regional-linear-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    model=ROOT/'models/qwen2.5-0.5b-7ae5576/model.safetensors';trace=ROOT/'evidence/reference-f32-greedy-20260908T0554/arithmetic/step-000.npz'
    state=load_file(model);reference=np.load(trace)
    weights=np.stack([state[f'model.layers.{layer}.mlp.up_proj.weight'][:128].float().numpy() for layer in (0,12)])
    initial=np.stack([reference[f'model.layers.{layer}.post_attention_layernorm'][0,-1] for layer in (0,12)])
    alternate=np.zeros_like(initial);alternate[0,895]=1;alternate[1,0]=-1
    batches=np.stack([initial,alternate]);sequences=[];expected=[]
    for inputs in batches:
        epochs=[];outputs=[];x=inputs.copy()
        for epoch in range(4):
            epochs.append(x.copy());outputs.append(np.stack([weights[g].astype(np.float64)@x[g].astype(np.float64) for g in range(2)]))
            x=(x+np.array([[.03125],[-.0625]],np.float32)).astype(np.float32)
        sequences.append(np.stack(epochs));expected.append(np.stack(outputs))
    packed=np.stack([np.stack([w[:,i*112:(i+1)*112].T.copy() for i in range(8)]) for w in weights]);packed=(packed.view(np.uint32)>>16).astype(np.uint16)
    np.savez(out/'inputs.npz',weights=weights,packed_weights=packed,inputs=batches,sequence=np.stack(sequences),expected=np.stack(expected))
    for src,dst in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/kernels/local_gemv_bf16_f32_colmajor.csl','kernel.csl'),(ROOT/'csl/runtime/line_allreduce_f32.csl','line.csl')]:shutil.copy2(src,out/dst)
    (out/'pe.csl').write_text('''param memcpy_params;param group:i16=-1;param ordinal:u16=0;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const kernel=@import_module("kernel.csl",.{.rows=128,.columns=112});
const line=@import_module(if(group>=0) "line.csl" else "<empty>",if(group>=0) .{.ordinal=ordinal,.participants=8,.length=128,.colors=[3]color{@get_color(0),@get_color(1),@get_color(2)},.task_id=@get_local_task_id(10),.on_complete=finished} else .{});
var weights=@zeros([14336]u16);var input=@zeros([112]f32);var partial=@zeros([128]f32);var result=@zeros([128]f32);var history=@constants([512]f32,-777.0);var progress=@zeros([1]u32);
fn reset() void {progress[0]=0;const d=@get_dsd(mem1d_dsd,.{.base_address=&history,.extent=512});@fmovs(d,-777.0);sys.unblock_cmd_stream();}
fn step() void {kernel.compute(&weights,&input,&partial);if(comptime group>=0){line.start(&partial,&result);}}
fn compute() void {if(comptime group>=0){@assert(progress[0]==0);step();}else{sys.unblock_cmd_stream();}}
fn finished() void {
 const src=@get_dsd(mem1d_dsd,.{.base_address=&result,.extent=128});const base=@get_dsd(mem1d_dsd,.{.base_address=&history,.extent=128});const dst=@increment_dsd_offset(base,@as(i16,progress[0]*128),f32);@mov32(dst,src);
 progress[0]+=1;
 if(progress[0]<4){const x=@get_dsd(mem1d_dsd,.{.base_address=&input,.extent=112});@fadds(x,x,if(group==0) @as(f32,0.03125) else @as(f32,-0.0625));step();}
 else{sys.unblock_cmd_stream();}
}
const wp:[*]u16=&weights;const xp:[*]f32=&input;const hp:[*]f32=&history;const pp:[*]u32=&progress;
comptime{@export_symbol(wp,"weights");@export_symbol(xp,"input");@export_symbol(hp,"history");@export_symbol(pp,"progress");@export_symbol(reset);@export_symbol(compute);}
''')
    (out/'layout.csl').write_text('''const memcpy=@import_module("<memcpy/get_params>",.{.width=22,.height=3});layout{@set_rectangle(22,3);
 for(@range(i16,22))|x|{for(@range(i16,3))|y|{
 const group:i16=if(y==1 and x>=2 and x<10) 0 else if(y==1 and x>=12 and x<20) 1 else -1;
 const ordinal:u16=if(group==0) @as(u16,x-2) else if(group==1) @as(u16,x-12) else 0;
 @set_tile_code(x,y,"pe.csl",.{.memcpy_params=memcpy.get_params(x),.group=group,.ordinal=ordinal});}}
 @export_name("weights",[*]u16,false);@export_name("input",[*]f32,false);@export_name("history",[*]f32,false);@export_name("progress",[*]u32,false);@export_name("reset",fn()void);@export_name("compute",fn()void);}
''')
    def sha(p):
        with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
    (out/'config.json').write_text(json.dumps(dict(model_sha256=sha(model),trace_sha256=sha(trace),origins=[[2,1],[12,1]],layers=[0,12],device_epochs=4,reset_sequences=2,gate=dict(relative_l2=2e-6,relative_peak=3e-6),scope='Two independent offset horizontal eight-PE real128x896 contractions, four on-device epochs per host launch, reset and changed input. Static route/queue ownership; not arbitrary routing or full-model qualification.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify,sha
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);c=json.loads((root/'config.json').read_text());data=np.load(root/'inputs.npz');cmd=['cslc','layout.csl','--arch=wse3','--fabric-dims=29,5','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');subprocess.run(cmd,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['weights','input','history','progress']};runner.load();runner.run();rows=[]
    def read(name,x,y,width,n):
        value=np.zeros(width*n,np.float32 if name=='history' else np.uint32);runner.memcpy_d2h(value,ids[name],x,y,width,1,n,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);return value.reshape(width,n)
    try:
        for g,(x,y) in enumerate(c['origins']):runner.memcpy_h2d(ids['weights'],data['packed_weights'][g].astype(np.uint32).ravel(),x,y,8,1,14336,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        for sequence in range(2):
            runner.launch('reset',nonblock=False)
            for g,(x,y) in enumerate(c['origins']):runner.memcpy_h2d(ids['input'],data['inputs'][sequence,g].copy(),x,y,8,1,112,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            runner.launch('compute',nonblock=False);actual={};checks=[]
            for g,(x,y) in enumerate(c['origins']):
                got=read('history',x,y,8,512).reshape(8,4,128);progress=read('progress',x,y,8,1);actual[f'group{g}']=got;actual[f'progress{g}']=progress;np.savez(root/f'actual-{sequence}.npz',**actual);expected=data['expected'][sequence,:,g];error=got.astype(np.float64)-expected;assert np.isfinite(got).all() and np.all(progress==4)
                l2=float(np.max(np.linalg.norm(error,axis=-1)/np.maximum(np.linalg.norm(expected,axis=-1),1e-30)));peak=float(np.max(np.max(np.abs(error),axis=-1)/np.maximum(np.max(np.abs(expected),axis=-1),1e-30)));assert l2<=c['gate']['relative_l2'] and peak<=c['gate']['relative_peak'],(sequence,g,l2,peak);checks.append(dict(group=g,relative_l2=l2,relative_peak=peak))
            for x,y in [(1,1),(10,1),(11,1),(20,1),(2,0),(12,2)]:
                history=read('history',x,y,1,512);progress=read('progress',x,y,1,1);actual[f'guard{x}-{y}']=history;actual[f'guard-progress{x}-{y}']=progress;np.savez(root/f'actual-{sequence}.npz',**actual);assert np.all(history==-777.0) and np.all(progress==0)
            np.savez(root/f'actual-{sequence}.npz',**actual);rows.append(dict(sequence=sequence,checks=checks,actual_sha256=sha(root/f'actual-{sequence}.npz')));print('REGIONAL SEQUENCE',sequence,flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=rows,scope=c['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

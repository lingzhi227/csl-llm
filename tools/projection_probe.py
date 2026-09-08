"""Native-dimension Qwen projection using resident BF16 tiles and dataflow sum."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(projection):
    import numpy as np
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0,str(ROOT/'src'))
    from csl_llm.projection_regions import Region,layout
    out=ROOT/'evidence'/('projection-'+projection+'-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    model=ROOT/'models/qwen2.5-0.5b-7ae5576/model.safetensors';trace=ROOT/'evidence/reference-f32/arithmetic/step-000.npz';state=load_file(model);ref=np.load(trace)
    tensor=f'model.layers.0.mlp.{projection}_proj.weight';weights=state[tensor].float().numpy();outputs,inner=weights.shape;groups=outputs//128;shards=(inner+111)//112
    if projection=='down':
        x=(torch.nn.functional.silu(torch.from_numpy(ref['model.layers.0.mlp.gate_proj'][0,-1]))*torch.from_numpy(ref['model.layers.0.mlp.up_proj'][0,-1])).numpy();width,height=22,groups*2;regions=[Region(0,2*g,22,2) for g in range(groups)]
    else:
        x=ref['model.layers.0.post_attention_layernorm'][0,-1].copy();width,height=16,(groups+1)//2;regions=[Region((g%2)*8,g//2,8,1) for g in range(groups)]
    packed=np.zeros((height,width,112,128),dtype=np.uint16);inputs=np.zeros((4,height,width,112),np.float32)
    cases=[x,np.eye(1,inner,inner-1,dtype=np.float32).ravel(),np.zeros(inner,np.float32),x.copy()]
    for g,region in enumerate(regions):
        for ordinal in range(shards):
            px,py=region.point(ordinal)
            # Match the candidate's ascending shard order in each physical row.
            shard=ordinal if shards==8 or ordinal<22 else 65-ordinal
            valid=min(112,inner-shard*112)
            tile=weights[g*128:(g+1)*128,shard*112:shard*112+valid].T.copy()
            packed[py,px,:valid]=(tile.view(np.uint32)>>16).astype(np.uint16)
            for case,vector in enumerate(cases):inputs[case,py,px,:valid]=vector[shard*112:shard*112+valid]
    expected=np.stack([weights.astype(np.float64)@v.astype(np.float64) for v in cases])
    official=ref[f'model.layers.0.mlp.{projection}_proj'][0,-1].copy()
    np.savez(out/'inputs.npz',packed_weights=packed,inputs=inputs,expected=expected,official=official)
    for src,dst in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/kernels/local_gemv_bf16_f32_colmajor.csl','kernel.csl'),(ROOT/'csl/runtime/line_projection_f32.csl','line.csl'),(ROOT/'configs/precision.json','precision.json'),(ROOT/'src/csl_llm/projection_regions.py','regions.py')]:shutil.copy2(src,out/dst)
    (out/'pe.csl').write_text('''param memcpy_params;param group:i16;param ordinal:u16;param participants:u16;param negative:direction;param positive:direction;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const clock=@import_module("<time>");const kernel=@import_module("kernel.csl",.{.rows=128,.columns=112});
const line=@import_module("line.csl",.{.ordinal=ordinal,.participants=participants,.length=128,.negative=negative,.positive=positive,.colors=[3]color{@get_color(0),@get_color(1),@get_color(2)},.task_id=@get_local_task_id(10),.on_complete=finished});
var weights=@zeros([14336]u16);var input=@zeros([112]f32);var partial=@zeros([128]f32);var result=@zeros([128]f32);var progress=@zeros([1]u32);var timing=@zeros([6]u16);var started=@zeros([3]u16);var ended=@zeros([3]u16);
fn reset() void {progress[0]=0;sys.unblock_cmd_stream();}
fn compute() void {clock.enable_tsc();clock.get_timestamp(&started);kernel.compute(&weights,&input,&partial);line.start(&partial,&result);}
fn finished() void {clock.get_timestamp(&ended);clock.disable_tsc();for(@range(u16,3))|i|{timing[i]=started[i];timing[i+3]=ended[i];}progress[0]+=1;sys.unblock_cmd_stream();}
const wp:[*]u16=&weights;const xp:[*]f32=&input;const rp:[*]f32=&result;const pp:[*]u32=&progress;const tp:[*]u16=&timing;
comptime{@export_symbol(wp,"weights");@export_symbol(xp,"input");@export_symbol(rp,"result");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(reset);@export_symbol(compute);}
''')
    (out/'layout.csl').write_text(layout(width,height,regions,[('weights','u16'),('input','f32'),('result','f32'),('progress','u32'),('timing','u16')],group_identity=False))
    (out/'config.json').write_text(json.dumps(dict(projection=projection,tensor=tensor,shape=[outputs,inner],width=width,height=height,roots=[list(r.point(0)) for r in regions],model_sha256=digest(model),trace_sha256=digest(trace),cases=['canonical','last-feature-onehot','zero','canonical-repeat'],gate=dict(relative_l2=2e-6,relative_peak=3e-6),scope='Complete native-dimension layer0 matrix projection, all output roots and all PE completion read, resident weights across four calls; not whole FFN/layer/model.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);c=json.loads((root/'config.json').read_text());data=np.load(root/'inputs.npz');w,h=c['width'],c['height'];policy=json.loads((root/'precision.json').read_text())['stage_gate']['absolute_or_normwise']
    def stage(name):(root/'stage.json').write_text(json.dumps(dict(stage=name))+'\n');print(name,flush=True)
    cmd=['cslc','layout.csl','--arch=wse3',f'--fabric-dims={w+7},{h+2}','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');stage('compile');subprocess.run(cmd,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['weights','input','result','progress','timing']};runner.load();runner.run();reports=[]
    def read(name,x,y,rw,rh,n):
        v=np.zeros(rw*rh*n,np.float32 if name=='result' else np.uint32);runner.memcpy_d2h(v,ids[name],x,y,rw,rh,n,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT if name=='timing' else MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);return v
    try:
        stage('load-resident-weights');runner.memcpy_h2d(ids['weights'],data['packed_weights'].astype(np.uint32).ravel(),0,0,w,h,14336,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        for epoch,name in enumerate(c['cases']):
            stage(name);runner.memcpy_h2d(ids['input'],data['inputs'][epoch].copy().ravel(),0,0,w,h,112,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);runner.launch('compute',nonblock=False);outputs=[];timing=[]
            for x,y in c['roots']:
                outputs.append(read('result',x,y,1,1,128));timing.append(read('timing',x,y,1,1,6))
                np.savez(root/f'partial-read-{epoch}.npz',outputs=np.stack(outputs),timing=np.stack(timing))
            actual=np.concatenate(outputs);progress=read('progress',0,0,w,h,1);timing=np.stack(timing);np.savez(root/f'actual-{epoch}.npz',output=actual,progress=progress,timing=timing)
            assert np.isfinite(actual).all() and np.all(progress==epoch+1);expected=data['expected'][epoch];error=actual.astype(np.float64)-expected;l2=float(np.linalg.norm(error)/max(np.linalg.norm(expected),1e-30));peak=float(np.max(np.abs(error))/max(np.max(np.abs(expected)),1e-30));assert l2<=c['gate']['relative_l2'] and peak<=c['gate']['relative_peak'],(name,l2,peak)
            official_check=None
            if epoch in (0,3):
                target=data['official'].astype(np.float64);e=actual.astype(np.float64)-target;absolute=float(np.max(np.abs(e)));norm=float(np.linalg.norm(e)/max(np.linalg.norm(target),1e-30));relative_peak=float(absolute/max(np.max(np.abs(target)),1e-30));assert absolute<=policy['max_abs'] or (norm<=policy['both']['relative_l2'] and relative_peak<=policy['both']['relative_peak']);official_check=dict(max_abs=absolute,relative_l2=norm,relative_peak=relative_peak)
            t=timing.astype(np.uint64);assert np.all(t<65536);ticks=lambda a:a[:,0]+(a[:,1]<<16)+(a[:,2]<<32);cycles=(ticks(t[:,3:])-ticks(t[:,:3]))&np.uint64((1<<48)-1);assert np.all((cycles>0)&(cycles<1<<40));reports.append(dict(name=name,relative_l2=l2,relative_peak=peak,official=official_check,root_cycles=cycles.tolist(),actual_sha256=digest(root/f'actual-{epoch}.npz')));(root/'results.json').write_text(json.dumps(dict(success=False,cases=reports),indent=2)+'\n')
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=reports,scope=c['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--projection',choices=['up'],default='up');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare(a.projection)
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

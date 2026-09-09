"""Actual Qwen 128x896 contraction across eight PEs and SDK collectives."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def prepare(calls=10):
    import numpy as np
    from safetensors.torch import load_file
    out=ROOT/'evidence'/('distributed-linear-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    source=ROOT/'models/qwen2.5-0.5b-7ae5576/model.safetensors';reference=ROOT/'evidence/reference-f32-greedy-20260908T0554/arithmetic'
    weight=load_file(source)['model.layers.0.mlp.up_proj.weight'][:128].float().numpy();inputs=[np.load(reference/f'step-{i:03d}.npz')['model.layers.0.post_attention_layernorm'][0,-1] for i in range(3)]
    for index in (0,111,112,895):
        x=np.zeros(896,np.float32);x[index]=1;inputs.append(x)
    inputs.extend([np.zeros(896,np.float32),-inputs[0],inputs[0].copy()]);inputs=np.stack(inputs)
    assert 1<=calls<=len(inputs)
    inputs=inputs[:calls]
    shards=np.stack([weight[:,i*112:(i+1)*112].T.copy() for i in range(8)])
    np.savez(out/'inputs.npz',weights=weight,packed_weights_u16=(shards.view(np.uint32)>>16).astype(np.uint16),inputs=inputs,expected=inputs.astype(np.float64)@weight.astype(np.float64).T,expected_partial=np.stack([inputs[:,i*112:(i+1)*112].astype(np.float64)@weight[:,i*112:(i+1)*112].astype(np.float64).T for i in range(8)],axis=1))
    shutil.copy2(__file__,out/'driver.py');shutil.copy2(ROOT/'tools/sdk_probe.py',out/'executor.py');shutil.copy2(ROOT/'csl/kernels/local_gemv_bf16_f32_colmajor.csl',out/'kernel.csl');shutil.copy2(ROOT/'csl/runtime/distributed_linear_pe.csl',out/'pe.csl')
    (out/'layout.csl').write_text('''const memcpy=@import_module("<memcpy/get_params>",.{.width=8,.height=1});const c2d=@import_module("<collectives_2d/params>");
layout{@set_rectangle(8,1);for(@range(i16,8))|x|{
 @set_tile_code(x,0,"pe.csl",.{.memcpy_params=memcpy.get_params(x),.c2d_params=c2d.get_params(@as(u16,x),0,.{.x_colors=.{@get_color(0),@get_color(1)},.x_entrypoints=.{@get_local_task_id(14),@get_local_task_id(15)},.y_colors=.{@get_color(4),@get_color(5)},.y_entrypoints=.{@get_local_task_id(16),@get_local_task_id(17)}})});
 }
 @export_name("weights",[*]u16,false);@export_name("input",[*]f32,false);@export_name("partial",[*]f32,false);@export_name("result",[*]f32,false);@export_name("progress",[*]u32,false);@export_name("timing",[*]u16,false);@export_name("initialize",fn()void);@export_name("compute",fn()void);
}
''')
    with source.open('rb') as f:model_sha=hashlib.file_digest(f,'sha256').hexdigest()
    (out/'config.json').write_text(json.dumps(dict(calls=len(inputs),model_sha256=model_sha,reference_files={f'step-{i:03d}.npz':hashlib.sha256((reference/f'step-{i:03d}.npz').read_bytes()).hexdigest() for i in range(3)},weight_tensor='model.layers.0.mlp.up_proj.weight',weight_slice=[[0,128],[0,896]],gate=dict(relative_l2=2e-6,relative_peak=3e-6),resources=dict(matrix_dsr_banks=[3,4],sdk_x_entrypoints=[14,15],completion_task=10,x_colors=[0,1]),scope='Full hidden896 contraction for128 actual UP outputs, eight input shards, allreduce to every PE; not full layer/model.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify,sha
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);cmd=['cslc','layout.csl','--arch=wse3','--fabric-dims=15,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');subprocess.run(cmd,check=True)
    c=json.loads((root/'config.json').read_text());data=np.load(root/'inputs.npz');runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ('weights','input','partial','result','progress','timing')};runner.load();runner.run();rows=[]
    try:
        runner.launch('initialize',nonblock=False);print('LOAD WEIGHTS',flush=True)
        runner.memcpy_h2d(ids['weights'],data['packed_weights_u16'].astype(np.uint32).ravel(),0,0,8,1,14336,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
        for epoch,x in enumerate(data['inputs']):
            runner.memcpy_h2d(ids['input'],x.copy(),0,0,8,1,112,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);runner.launch('compute',nonblock=False);actual={}
            for name,n,bits in [('partial',128,32),('result',128,32),('progress',1,32),('timing',9,16)]:
                value=np.zeros(8*n,np.float32 if name in ('partial','result') else np.uint32);runner.memcpy_d2h(value,ids[name],0,0,8,1,n,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT if bits==16 else MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);actual[name]=value.reshape(8,n)
            np.savez(root/f'actual-{epoch}.npz',**actual);checks={}
            for name,expected in [('partial',data['expected_partial'][epoch]),('result',np.tile(data['expected'][epoch],(8,1)))]:
                got=actual[name].astype(np.float64);assert np.isfinite(got).all();error=got-expected;l2=float(np.max(np.linalg.norm(error,axis=1)/np.maximum(np.linalg.norm(expected,axis=1),1e-30)));peak=float(np.max(np.max(np.abs(error),axis=1)/np.maximum(np.max(np.abs(expected),axis=1),1e-30)));assert l2<=c['gate']['relative_l2'] and peak<=c['gate']['relative_peak'],(name,l2,peak);checks[name]=dict(relative_l2=l2,relative_peak=peak)
            assert np.all(actual['progress']==epoch+1);t=actual['timing'].astype(np.uint64);assert np.all(t<65536);ticks=lambda v:v[:,0]+(v[:,1]<<16)+(v[:,2]<<32);cycles=(ticks(t[:,6:])-ticks(t[:,:3]))&np.uint64((1<<48)-1);assert np.all((cycles>0)&(cycles<1<<40));rows.append(dict(epoch=epoch,checks=checks,cycles=cycles.tolist(),kernel_cycles=((ticks(t[:,3:6])-ticks(t[:,:3]))&np.uint64((1<<48)-1)).tolist(),collective_cycles=((ticks(t[:,6:])-ticks(t[:,3:6]))&np.uint64((1<<48)-1)).tolist(),actual_sha256=sha(root/f'actual-{epoch}.npz')));print('DISTRIBUTED CALL',epoch+1,flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=rows,runtime_instances=1,inputs_sha256=sha(root/'inputs.npz')),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--calls',type=int,default=10);p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare(a.calls)
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

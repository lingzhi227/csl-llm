"""Full2048-position striped KV capacity with device-generated diagnostic inputs.

Uses real projection vectors as seeds, but is not a complete model trajectory.
The position-dependent generation and FP32 rounding order are fixed below.
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    import numpy as np
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding,apply_rotary_pos_emb
    sys.path.insert(0,str(ROOT/'src'))
    from csl_llm.regions import Region,layout
    out=ROOT/'evidence'/('gqa-capacity-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    trace=ROOT/'evidence/reference-f32-greedy-20260908T0554/arithmetic/step-000.npz';model=ROOT/'models/qwen2.5-0.5b-7ae5576';r=np.load(trace)
    rotary=Qwen2RotaryEmbedding(AutoConfig.from_pretrained(model,local_files_only=True));frequency=rotary.inv_freq.numpy().copy()
    counts=[2048,65,1];layers=[0,12,23];queries=[];keys=[];values=[];cache_keys=[];cache_values=[];expected=[]
    for count,layer in zip(counts,layers):
        prefix=f'model.layers.{layer}.self_attn.'
        q=r[prefix+'q_proj'][0,-1].reshape(14,64).copy();k=r[prefix+'k_proj'][0,-1].reshape(2,64).copy();v=r[prefix+'v_proj'][0,-1].reshape(2,64).copy()
        positions=torch.arange(count).reshape(1,-1)
        qt=torch.from_numpy(q).reshape(1,14,1,64).expand(1,14,count,64);kt=torch.from_numpy(k).reshape(1,2,1,64).expand(1,2,count,64)
        cosine,sine=rotary(qt,positions);rotated_q,rotated_k=apply_rotary_pos_emb(qt,kt,cosine,sine)
        key_cache=rotated_k.numpy()[0].transpose(1,0,2).copy()
        # Multiply rounds to FP32, then addition rounds separately, matching CSL.
        scale=(np.float32(1)+np.arange(count,dtype=np.float32)*np.float32(1/2048)).astype(np.float32)
        offset=((np.arange(count)%7).astype(np.float32)*np.float32(1/32)).astype(np.float32)
        value_cache=((v[None]*scale[:,None,None]).astype(np.float32)+offset[:,None,None]).astype(np.float32)
        contexts=[]
        for h in range(14):
            score=key_cache[:,h//7].astype(np.float64)@rotated_q.numpy()[0,h,-1].astype(np.float64)/8
            p=np.exp(score-np.max(score));p/=np.sum(p);contexts.append(p@value_cache[:,h//7].astype(np.float64))
        padded_k=np.zeros((2048,2,64),np.float32);padded_v=np.zeros_like(padded_k);padded_k[:count]=key_cache;padded_v[:count]=value_cache
        queries.append(q);keys.append(k);values.append(v);cache_keys.append(padded_k);cache_values.append(padded_v);expected.append(np.stack(contexts))
    np.savez(out/'inputs.npz',query=np.stack(queries),key=np.stack(keys),value=np.stack(values),frequency=frequency,expected=np.stack(expected),expected_cache_k=np.stack(cache_keys),expected_cache_v=np.stack(cache_values))
    for src,dst in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/kernels/kv_block_f32.csl','kv.csl'),(ROOT/'csl/kernels/qwen_position_f32.csl','rope.csl'),(ROOT/'csl/runtime/line_allreduce_f32.csl','line.csl'),(ROOT/'configs/precision.json','precision.json'),(ROOT/'src/csl_llm/regions.py','regions.py')]:shutil.copy2(src,out/dst)
    (out/'pe.csl').write_text('''param memcpy_params;param group:i16=-1;param ordinal:u16=0;param participants:u16;param negative:direction;param positive:direction;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const kv=@import_module("kv.csl",.{.stripe=ordinal});const qr=@import_module("rope.csl",.{.heads=7});const kr=@import_module("rope.csl",.{.heads=1});
const line=@import_module(if(group>=0) "line.csl" else "<empty>",if(group>=0) .{.ordinal=ordinal,.participants=participants,.length=455,.negative=negative,.positive=positive,.colors=[3]color{@get_color(0),@get_color(1),@get_color(2)},.task_id=@get_local_task_id(10),.on_complete=continued} else .{});
var query=@zeros([448]f32);var key=@zeros([64]f32);var value=@zeros([64]f32);var frequency=@zeros([32]f32);var key_rotated=@zeros([64]f32);var value_scaled=@zeros([64]f32);
var local=@zeros([455]f32);var global=@zeros([455]f32);var context=@zeros([448]f32);var control=@zeros([1]u32);var progress=@zeros([1]u32);var status=@zeros([2]u32);var phase:u16=0;
fn reset() void {kv.reset();progress[0]=0;status[0]=0;status[1]=0;sys.unblock_cmd_stream();}
fn compute() void {
 if(comptime group>=0){
  const count=@as(u16,control[0]);@assert(count>0 and count<=2048 and kv.seen==0);
  for(@range(u16,2048))|position|{if(position<count){
   if(position/32==ordinal){
    kr.prepare(position,&frequency);kr.apply(&key,&key_rotated);
    const source=@get_dsd(mem1d_dsd,.{.base_address=&value,.extent=64});const dst=@get_dsd(mem1d_dsd,.{.base_address=&value_scaled,.extent=64});
    const scale:f32=1.0+@as(f32,position)*0.00048828125;const offset:f32=@as(f32,position%7)*0.03125;
    @fmuls(dst,source,scale);@fadds(dst,dst,offset);
   }
   kv.append(position,&key_rotated,&value_scaled);
  }}
  qr.prepare(count-1,&frequency);qr.apply(&query,&query);kv.peaks(&query,count-1,&local);phase=0;line.start_count(&local,&global,7,true);
 }else{sys.unblock_cmd_stream();}
}
fn continued() void {
 if(comptime group>=0){if(phase==0){phase=1;kv.statistics(&global,&local);line.start_count(&local,&global,455,false);}
 else{@assert(phase==1);kv.normalize(&global,&context);progress[0]+=1;status[0]=@as(u32,kv.seen);status[1]=@as(u32,kv.valid);sys.unblock_cmd_stream();}}
}
const qp:[*]f32=&query;const kp:[*]f32=&key;const vp:[*]f32=&value;const fp:[*]f32=&frequency;const cp:[*]f32=&context;const ctrl:[*]u32=&control;const pp:[*]u32=&progress;const sp:[*]u32=&status;const cache_k:[*]f32=&kv.keys;const cache_v:[*]f32=&kv.values;
comptime{@export_symbol(qp,"query");@export_symbol(kp,"key");@export_symbol(vp,"value");@export_symbol(fp,"frequency");@export_symbol(cp,"context");@export_symbol(ctrl,"control");@export_symbol(pp,"progress");@export_symbol(sp,"status");@export_symbol(cache_k,"cache_k");@export_symbol(cache_v,"cache_v");@export_symbol(reset);@export_symbol(compute);}
''')
    regions=[Region(1,1,11,6),Region(14,1,11,6)]
    exports=[('query','f32'),('key','f32'),('value','f32'),('frequency','f32'),('context','f32'),('control','u32'),('progress','u32'),('status','u32'),('cache_k','f32'),('cache_v','f32')]
    (out/'layout.csl').write_text(layout(26,8,regions,exports))
    (out/'config.json').write_text(json.dumps(dict(counts=counts,layers=layers,origins=[[1,1],[14,1]],trace_sha256=digest(trace),model_config_sha256=digest(model/'config.json'),cache_observed_stripes=[0,2,63],formula='Q,K: fixed actual projection seeds, Qwen FP32 half-split RoPE at each position. V: float32(float32(seed * float32(1 + position/2048)) + float32((position%7)/32)); separate multiply/add rounding.',scope='Full2048 KV capacity,66-stripe serpentine regions per KV head (last2 empty),14Q/2KV, reset to65 then1. Synthetic diagnostic trajectory from real seeds. All stripe counts read; cache contents sampled in three stripes per head. Not2048-token actual model inference.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);c=json.loads((root/'config.json').read_text());data=np.load(root/'inputs.npz');gate=json.loads((root/'precision.json').read_text())['stage_gate']['absolute_or_normwise']
    def stage(name):(root/'stage.json').write_text(json.dumps(dict(stage=name))+'\n');print(name,flush=True)
    cmd=['cslc','layout.csl','--arch=wse3','--fabric-dims=33,10','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');stage('compile');subprocess.run(cmd,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['query','key','value','frequency','control','context','progress','status','cache_k','cache_v']};runner.load();runner.run();reports=[]
    def read(name,x,y,width,height,n):
        v=np.zeros(width*height*n,np.uint32 if name in ('status','progress') else np.float32);runner.memcpy_d2h(v,ids[name],x,y,width,height,n,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);return v.reshape(height,width,n)
    def check(got,expected,label):
        got=got.astype(np.float64);expected=expected.astype(np.float64);assert np.isfinite(got).all();error=got-expected;absolute=float(np.max(np.abs(error))) if error.size else 0.0;l2=float(np.linalg.norm(error)/max(np.linalg.norm(expected),1e-30));peak=float(absolute/max(float(np.max(np.abs(expected))) if expected.size else 0.,1e-30));assert absolute<=gate['max_abs'] or (l2<=gate['both']['relative_l2'] and peak<=gate['both']['relative_peak']),(label,absolute,l2,peak);return dict(max_abs=absolute,relative_l2=l2,relative_peak=peak)
    try:
        for sequence,count in enumerate(c['counts']):
            stage(f'reset-{count}');runner.launch('reset',nonblock=False)
            for group,(x,y) in enumerate(c['origins']):
                for name,value in [('query',data['query'][sequence,group*7:(group+1)*7].ravel()),('key',data['key'][sequence,group]),('value',data['value'][sequence,group]),('frequency',data['frequency']),('control',np.array([count],np.uint32))]:
                    runner.memcpy_h2d(ids[name],np.tile(value,66),x,y,11,6,len(value),streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            stage(f'compute-{count}');runner.launch('compute',nonblock=False);actual={};checks=[]
            for group,(x,y) in enumerate(c['origins']):
                stage(f'context-{count}-head{group}');actual[f'context{group}']=read('context',x,y,1,1,448).reshape(7,64);actual[f'status{group}']=read('status',x,y,11,6,2);actual[f'progress{group}']=read('progress',x,y,11,6,1);np.savez(root/f'actual-{sequence}.npz',**actual)
                checks.append(dict(group=group,context=check(actual[f'context{group}'],data['expected'][sequence,group*7:(group+1)*7],('context',count,group))))
                assert np.all(actual[f'progress{group}']==1)
                for ordinal in range(66):
                    row,col=divmod(ordinal,11);col=col if row%2==0 else 10-col;valid=min(32,max(0,count-ordinal*32));assert np.array_equal(actual[f'status{group}'][row,col],np.array([count,valid],np.uint32))
                for ordinal in c['cache_observed_stripes']:
                    row,col=divmod(ordinal,11);col=col if row%2==0 else 10-col;prefix=f'g{group}-stripe{ordinal}';actual[prefix+'-key']=read('cache_k',x+col,y+row,1,1,2048).reshape(64,32);actual[prefix+'-value']=read('cache_v',x+col,y+row,1,1,2048).reshape(32,64);np.savez(root/f'actual-{sequence}.npz',**actual)
                    valid=min(32,max(0,count-ordinal*32));start=ordinal*32;check(actual[prefix+'-key'][:,:valid].T,data['expected_cache_k'][sequence,start:start+valid,group],('key',count,group,ordinal));assert np.array_equal(actual[prefix+'-value'][:valid],data['expected_cache_v'][sequence,start:start+valid,group])
            reports.append(dict(count=count,checks=checks,actual_sha256=digest(root/f'actual-{sequence}.npz')));(root/'results.json').write_text(json.dumps(dict(success=False,cases=reports),indent=2)+'\n')
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=reports,scope=c['scope']),indent=2)+'\n')


def compile_preflight(root):
    sys.path.insert(0,str(root))
    from executor import verify
    verify(root);os.chdir(root)
    assert not (root/'preflight').exists()
    command=['cslc','layout.csl','--arch=wse3','--fabric-dims=33,10','--fabric-offsets=4,1','-o=preflight','--memcpy','--channels=1','--dump-dsr-alloc-graph']
    (root/'preflight-command.json').write_text(json.dumps(command)+'\n')
    subprocess.run(command,check=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--compile-only',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    elif a.compile_only:compile_preflight(a.compile_only.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

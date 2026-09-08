"""Persistent striped causal GQA, using real pinned Qwen Q/K/V traces.

Host supplies operator-test inputs only; all KV updates, score arithmetic,
regional softmax and context computation under test execute in CSL.
"""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare(stripes):
    import numpy as np
    import torch
    from transformers import AutoConfig
    from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding,apply_rotary_pos_emb
    assert stripes in (4,66)
    out=ROOT/'evidence'/('causal-gqa-'+str(stripes)+'stripes-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    reference=ROOT/'evidence/reference-f32-greedy-20260908T0554/arithmetic';model=ROOT/'models/qwen2.5-0.5b-7ae5576'
    rotary=Qwen2RotaryEmbedding(AutoConfig.from_pretrained(model,local_files_only=True));queries=[];keys=[];values=[];official=[];bindings={}
    for step in range(5):
        file=reference/f'step-{step:03d}.npz';bindings[file.name]=digest(file);r=np.load(file)
        q=r['model.layers.0.self_attn.q_proj'].reshape(1,-1,14,64).transpose(0,2,1,3).copy();k=r['model.layers.0.self_attn.k_proj'].reshape(1,-1,2,64).transpose(0,2,1,3).copy()
        positions=torch.arange(len(queries),len(queries)+q.shape[2]).reshape(1,-1)
        cosine,sine=rotary(torch.from_numpy(q),positions);rotated_q,rotated_k=apply_rotary_pos_emb(torch.from_numpy(q),torch.from_numpy(k),cosine,sine)
        value=r['model.layers.0.self_attn.v_proj'].reshape(-1,2,64)
        cached_v=r['cache.0.value'][0]
        probability=r['attention.0.probability'][0]
        for t in range(q.shape[2]):
            queries.append(rotated_q.numpy()[0,:,t]);keys.append(rotated_k.numpy()[0,:,t]);values.append(value[t])
            official.append(np.stack([probability[h,t].astype(np.float64)@cached_v[h//7].astype(np.float64) for h in range(14)]))
    q=np.stack(queries).astype(np.float32);k=np.stack(keys).astype(np.float32);v=np.stack(values).astype(np.float32);assert len(q)==34
    all_q=np.stack([q,np.concatenate([q[:3]*np.float32(-.75),np.zeros_like(q[3:])])]);all_k=np.stack([k,np.concatenate([k[:3]*np.float32(.5),np.zeros_like(k[3:])])]);all_v=np.stack([v,np.concatenate([v[:3]+np.float32(1),np.zeros_like(v[3:])])])
    expected=np.zeros((2,34,14,64),np.float64)
    for sequence,count in enumerate([34,3]):
        for position in range(count):
            for head in range(14):
                score=all_k[sequence,:position+1,head//7].astype(np.float64)@all_q[sequence,position,head].astype(np.float64)/8
                p=np.exp(score-np.max(score));p/=np.sum(p);expected[sequence,position,head]=p@all_v[sequence,:position+1,head//7].astype(np.float64)
    official=np.stack(official);assert np.max(np.abs(expected[0]-official))<1e-5
    np.savez(out/'inputs.npz',queries=all_q,keys=all_k,values=all_v,expected=expected,official_context=official)
    for src,dst in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/kernels/kv_block_f32.csl','kv.csl'),(ROOT/'csl/runtime/line_allreduce_f32.csl','line.csl'),(ROOT/'configs/precision.json','precision.json')]:shutil.copy2(src,out/dst)
    width=4 if stripes==4 else 11;height=stripes//width;app_width=2*(width+2);app_height=height+2;origins=[[1,1],[width+3,1]]
    (out/'pe.csl').write_text('''param memcpy_params;param group:i16=-1;param ordinal:u16=0;param participants:u16;param negative:direction;param positive:direction;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const kv=@import_module("kv.csl",.{.stripe=ordinal});
const line=@import_module(if(group>=0) "line.csl" else "<empty>",if(group>=0) .{.ordinal=ordinal,.participants=participants,.length=455,.negative=negative,.positive=positive,.colors=[3]color{@get_color(0),@get_color(1),@get_color(2)},.task_id=@get_local_task_id(10),.on_complete=continued} else .{});
var query=@zeros([448]f32);var key=@zeros([64]f32);var value=@zeros([64]f32);var local=@zeros([455]f32);var global=@zeros([455]f32);var context=@zeros([448]f32);var control=@zeros([1]u32);var progress=@zeros([1]u32);var status=@zeros([2]u32);var phase:u16=0;
fn reset() void {kv.reset();progress[0]=0;status[0]=0;status[1]=0;sys.unblock_cmd_stream();}
fn compute() void {
 if(comptime group>=0){
  const position=@as(u16,control[0]);kv.append(position,&key,&value);kv.peaks(&query,position,&local);phase=0;line.start_count(&local,&global,7,true);
 }else{sys.unblock_cmd_stream();}
}
fn continued() void {
 if(comptime group>=0){
  if(phase==0){phase=1;kv.statistics(&global,&local);line.start_count(&local,&global,455,false);}
  else{@assert(phase==1);kv.normalize(&global,&context);progress[0]+=1;status[0]=@as(u32,kv.seen);status[1]=@as(u32,kv.valid);sys.unblock_cmd_stream();}
 }
}
const qp:[*]f32=&query;const kp:[*]f32=&key;const vp:[*]f32=&value;const cp:[*]f32=&context;const ctrl:[*]u32=&control;const pp:[*]u32=&progress;const sp:[*]u32=&status;const cache_k:[*]f32=&kv.keys;const cache_v:[*]f32=&kv.values;const gp:[*]f32=&global;
comptime{@export_symbol(qp,"query");@export_symbol(kp,"key");@export_symbol(vp,"value");@export_symbol(cp,"context");@export_symbol(ctrl,"control");@export_symbol(pp,"progress");@export_symbol(sp,"status");@export_symbol(cache_k,"cache_k");@export_symbol(cache_v,"cache_v");@export_symbol(gp,"statistics");@export_symbol(reset);@export_symbol(compute);}
''')
    # Serpentine path uses each PE's actual previous/next physical neighbor.
    def point(ordinal,origin):
        row,col=divmod(ordinal,width);return (origin[0]+(col if row%2==0 else width-1-col),origin[1]+row)
    def direction(a,b):
        return {(1,0):'EAST',(-1,0):'WEST',(0,1):'SOUTH',(0,-1):'NORTH'}[(b[0]-a[0],b[1]-a[1])]
    tiles={}
    for group,origin in enumerate(origins):
        for ordinal in range(stripes):
            p=point(ordinal,origin);negative=direction(p,point(ordinal-1,origin)) if ordinal else 'WEST';positive=direction(p,point(ordinal+1,origin)) if ordinal<stripes-1 else 'EAST';tiles[p]=(group,ordinal,negative,positive)
    lines=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={app_width},.height={app_height}}});layout{{@set_rectangle({app_width},{app_height});']
    for y in range(app_height):
        for x in range(app_width):
            group,ordinal,negative,positive=tiles.get((x,y),(-1,0,'WEST','EAST'))
            lines.append(f'@set_tile_code({x},{y},"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),.group={group},.ordinal={ordinal},.participants={stripes},.negative={negative},.positive={positive}}});')
    for name,kind in [('query','f32'),('key','f32'),('value','f32'),('context','f32'),('control','u32'),('progress','u32'),('status','u32'),('cache_k','f32'),('cache_v','f32'),('statistics','f32')]:lines.append(f'@export_name("{name}",[*]{kind},false);')
    lines.append('@export_name("reset",fn()void);@export_name("compute",fn()void);}')
    (out/'layout.csl').write_text('\n'.join(lines)+'\n')
    (out/'config.json').write_text(json.dumps(dict(stripes=stripes,width=width,height=height,app_width=app_width,app_height=app_height,origins=origins,sequence_lengths=[34,3],checkpoints=[[0,29,30,31,32,33],[0,2]],trace_files=bindings,model_config_sha256=digest(model/'config.json'),scope='Actual14Q/2KV causal attention across32-token stripes,34 sequential cache appends and dirty-cache reset to3 changed tokens. Empty stripes use negative finite maximum sentinel and zero sum/context. Full declared context capacity and full layer/model remain unqualified.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);c=json.loads((root/'config.json').read_text());data=np.load(root/'inputs.npz');gate=json.loads((root/'precision.json').read_text())['stage_gate']['absolute_or_normwise']
    def stage(name):(root/'stage.json').write_text(json.dumps(dict(stage=name))+'\n');print(name,flush=True)
    cmd=['cslc','layout.csl','--arch=wse3',f"--fabric-dims={c['app_width']+7},{c['app_height']+2}",'--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');stage('compile');subprocess.run(cmd,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['query','key','value','control','context','progress','status','cache_k','cache_v','statistics']};runner.load();runner.run();rows=[]
    def read(name,x,y,n):
        v=np.zeros(n,np.uint32 if name in ('status','progress') else np.float32);runner.memcpy_d2h(v,ids[name],x,y,1,1,n,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);return v
    try:
        for sequence,count in enumerate(c['sequence_lengths']):
            runner.launch('reset',nonblock=False)
            for position in range(count):
                stage(f'sequence-{sequence}-append-{position}')
                for group,(x,y) in enumerate(c['origins']):
                    for name,value in [('query',data['queries'][sequence,position,group*7:(group+1)*7].ravel()),('key',data['keys'][sequence,position,group]),('value',data['values'][sequence,position,group]),('control',np.array([position],np.uint32))]:
                        tiled=np.tile(value,c['stripes']);runner.memcpy_h2d(ids[name],tiled,x,y,c['width'],c['height'],len(value),streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
                runner.launch('compute',nonblock=False)
                if position not in c['checkpoints'][sequence]:continue
                actual={};checks=[]
                for group,(x,y) in enumerate(c['origins']):
                    actual[f'context{group}']=read('context',x,y,448).reshape(7,64);actual[f'progress{group}']=read('progress',x,y,1);actual[f'statistics{group}']=read('statistics',x,y,455);np.savez(root/f'actual-{sequence}-{position}.npz',**actual)
                    got=actual[f'context{group}'].astype(np.float64);expected=data['expected'][sequence,position,group*7:(group+1)*7];assert np.isfinite(got).all();error=got-expected;absolute=float(np.max(np.abs(error)));l2=float(np.linalg.norm(error)/max(np.linalg.norm(expected),1e-30));peak=float(absolute/max(np.max(np.abs(expected)),1e-30));assert absolute<=gate['max_abs'] or (l2<=gate['both']['relative_l2'] and peak<=gate['both']['relative_peak']),(sequence,position,group,absolute,l2,peak);assert actual[f'progress{group}'][0]==position+1;assert np.all(actual[f'statistics{group}'][:7]>0);checks.append(dict(group=group,max_abs=absolute,relative_l2=l2,relative_peak=peak))
                rows.append(dict(sequence=sequence,position=position,checks=checks,actual_sha256=digest(root/f'actual-{sequence}-{position}.npz')));(root/'results.json').write_text(json.dumps(dict(success=False,cases=rows),indent=2)+'\n')
            cache={}
            for group,origin in enumerate(c['origins']):
                for ordinal in sorted(set([0,1,c['stripes']-1])):
                    row,col=divmod(ordinal,c['width']);x=origin[0]+(col if row%2==0 else c['width']-1-col);y=origin[1]+row
                    prefix=f'g{group}-stripe{ordinal}';cache[prefix+'-key']=read('cache_k',x,y,2048).reshape(64,32);cache[prefix+'-value']=read('cache_v',x,y,2048).reshape(32,64);cache[prefix+'-status']=read('status',x,y,2);np.savez(root/f'cache-{sequence}.npz',**cache)
                    valid=min(32,max(0,count-ordinal*32));assert np.array_equal(cache[prefix+'-status'],np.array([count,valid],np.uint32));assert np.array_equal(cache[prefix+'-key'][:,:valid].T,data['keys'][sequence,ordinal*32:ordinal*32+valid,group]);assert np.array_equal(cache[prefix+'-value'][:valid],data['values'][sequence,ordinal*32:ordinal*32+valid,group])
            rows.append(dict(sequence=sequence,cache_sha256=digest(root/f'cache-{sequence}.npz')))
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=rows,scope=c['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--stripes',type=int,default=4);p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare(a.stripes)
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

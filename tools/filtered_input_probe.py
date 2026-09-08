"""Counter-filter multicast of a full4864-feature FFN activation plus padding."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    import numpy as np
    import torch
    out=ROOT/'evidence'/('filtered-input-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    trace=ROOT/'evidence/reference-f32/arithmetic/step-000.npz';r=np.load(trace);x=(torch.nn.functional.silu(torch.from_numpy(r['model.layers.0.mlp.gate_proj'][0,-1]))*torch.from_numpy(r['model.layers.0.mlp.up_proj'][0,-1])).numpy()
    frame=np.zeros(4928,np.float32);frame[:4864]=x;frames=np.stack([frame,-frame,np.zeros_like(frame),frame]);np.save(out/'inputs.npy',frames)
    for src,dst in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/programs/filtered_input/input.csl','input.csl')]:shutil.copy2(src,out/dst)
    (out/'pe.csl').write_text('''param memcpy_params;param role:u16;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);const callback=@get_local_task_id(11);
const receiver=@import_module(if(role==1) "input.csl" else "<empty>",if(role==1) .{.length=112,.input_color=@get_color(3),.completion_task=callback} else .{});
const source_length=if(role==0) 4928 else 1;var frame=@zeros([source_length]f32);var input=@constants([112]f32,-777.0);var progress=@zeros([1]u32);
const oq=@get_output_queue(4);const dest=@get_dsr(dsr_dest,5);const source=@get_dsr(dsr_src1,5);
fn compute() void {
 if(comptime role==0){const mem=@get_dsd(mem1d_dsd,.{.base_address=&frame,.extent=4928});const out=@get_dsd(fabout_dsd,.{.output_queue=oq,.extent=4928});@load_to_dsr(dest,out,.{.async=true,.activate=callback});@load_to_dsr(source,mem);@mov32(dest,source,.{.async=true});}
 else if(comptime role==1){receiver.arm(&input);}else{sys.unblock_cmd_stream();}
}
task finished() void {progress[0]+=1;sys.unblock_cmd_stream();}
const fp:[*]f32=&frame;const ip:[*]f32=&input;const pp:[*]u32=&progress;
comptime{if(role==0){@initialize_queue(oq,.{.color=@get_color(3)});}@bind_local_task(finished,callback);@export_symbol(fp,"frame");@export_symbol(ip,"input");@export_symbol(pp,"progress");@export_symbol(compute);}
''')
    lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=23,.height=5});const flow_color=@get_color(3);layout{@set_rectangle(23,5);']
    for y in range(5):
        for x in range(23):
            role=0 if (x,y)==(0,0) else 1 if x>=1 and y>=1 else 2
            lines.append(f'@set_tile_code({x},{y},"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),.role=@as(u16,{role})}});')
            if (x,y)==(0,0):routes='.rx=.{RAMP},.tx=.{EAST}'
            elif y==0 and x>0:routes='.rx=.{WEST},.tx=.{SOUTH'+(',EAST' if x<22 else '')+'}'
            elif x>0:routes='.rx=.{NORTH},.tx=.{RAMP'+(',SOUTH' if y<4 else '')+'}'
            else:continue
            filter_text=''
            if role==1:
                shard=((y-1)%2)*22+x-1;counter=(4928-shard*112)%4928
                filter_text=f',.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter={counter},.max_counter=111,.limit1=4927}}'
            lines.append(f'@set_color_config({x},{y},flow_color,.{{.routes=.{{{routes}}}{filter_text}}});')
    lines.append('@export_name("frame",[*]f32,false);@export_name("input",[*]f32,false);@export_name("progress",[*]u32,false);@export_name("compute",fn()void);}')
    (out/'layout.csl').write_text('\n'.join(lines)+'\n')
    (out/'config.json').write_text(json.dumps(dict(trace_sha256=digest(trace),frame_length=4928,valid_features=4864,shards=44,replicas=2,cases=4,scope='Four fixed-period frames, two44-shard replicas, counter-filter112 elements perPE including64 padded tail values; no filter/queue reconfiguration. Not full layer or global multicast qualification.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);frames=np.load(root/'inputs.npy');cmd=['cslc','layout.csl','--arch=wse3','--fabric-dims=30,7','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');subprocess.run(cmd,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['frame','input','progress']};runner.load();runner.run();reports=[]
    try:
        for epoch,frame in enumerate(frames):
            runner.memcpy_h2d(ids['frame'],frame.copy(),0,0,1,1,4928,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);runner.launch('compute',nonblock=False);actual={}
            for name,n,dtype in [('input',112,np.float32),('progress',1,np.uint32)]:
                value=np.zeros(23*5*n,dtype);runner.memcpy_d2h(value,ids[name],0,0,23,5,n,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);actual[name]=value.reshape(5,23,n)
            np.savez(root/f'actual-{epoch}.npz',**actual)
            for y in range(5):
                for x in range(23):
                    if x>=1 and y>=1:
                        shard=((y-1)%2)*22+x-1;assert np.array_equal(actual['input'][y,x].view(np.uint32),frame[shard*112:(shard+1)*112].view(np.uint32));assert actual['progress'][y,x,0]==epoch+1
                    else:assert np.all(actual['input'][y,x]==-777.0) and actual['progress'][y,x,0]==(epoch+1 if (x,y)==(0,0) else 0)
            reports.append(dict(epoch=epoch,actual_sha256=digest(root/f'actual-{epoch}.npz')));print('FILTERED FRAME',epoch,flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=reports,scope=json.loads((root/'config.json').read_text())['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

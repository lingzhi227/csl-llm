"""Concurrent38-source packet fan-in using the planned FFN root geometry."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    out=ROOT/'evidence'/('packet-fanin-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    for source,name in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/runtime/packets.csl','packets.csl')]:shutil.copy2(source,out/name)
    (out/'pe.csl').write_text('''param memcpy_params;param source_id:i16;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const packets=@import_module("packets.csl",.{.sent_task_id=@get_local_task_id(12),.received_task_id=@get_local_task_id(13),.on_sent=sent,.on_received=received});
const is_root=source_id == -2;const history_size=if(is_root) 4864 else 1;
var history=@constants([history_size]u32,0xcafebabe);var counts=@zeros([38]u16);var tx=@zeros([31]u32);var control=@zeros([1]u32);var progress=@zeros([1]u32);
var offset:u16=0;var sending_count:u16=0;var completed:u16=0;var command_waiting:bool=false;
fn initialize() void {packets.init();sys.unblock_cmd_stream();}
fn reset() void {
 @assert(!packets.sending and !packets.receiving);offset=0;completed=0;progress[0]=0;command_waiting=false;
 for(@range(u16,38))|i|{counts[i]=0;}
 const d=@get_dsd(mem1d_dsd,.{.base_address=&history,.extent=history_size});@mov32(d,0xcafebabe);sys.unblock_cmd_stream();
}
fn value(group:u16,index:u16) u32 {return @as(u32,0xa5a55a5a)^(@as(u32,group)*0x01010101)^(@as(u32,index)<<12)^(control[0]<<24);}
fn transmit() void {
 if(comptime source_id>=0){
 @assert(source_id>=0 and offset<128);sending_count=if(128-offset<29) 128-offset else 29;
 tx[0]=(control[0]<<16)|@as(u32,source_id);tx[1]=@as(u32,offset);
 for(@range(u16,29))|i|{if(i<sending_count){tx[2+i]=value(@as(u16,source_id),offset+i);}}
 packets.send(62,0,&tx,sending_count+2);
 }
}
fn run() void {
 if(comptime source_id>=0){transmit();}
 else if(comptime is_root){command_waiting=true;finish_root();}
 else{sys.unblock_cmd_stream();}
}
fn finish_root() void {
 if(command_waiting and completed==38){progress[0]=38;command_waiting=false;sys.unblock_cmd_stream();}
}
fn sent() void {
 @assert(source_id>=0);offset+=sending_count;
 if(offset<128){transmit();}else{progress[0]=128;sys.unblock_cmd_stream();}
}
fn received(data:[*]u32,length:u16) void {
 @assert(is_root and length>=3);const group=@as(u16,data[0]&0xffff);const epoch=data[0]>>16;const begin=@as(u16,data[1]);const n=length-2;
 @assert(group<38 and epoch==control[0] and begin==counts[group] and begin+n<=128);
 if(comptime is_root){for(@range(u16,29))|i|{if(i<n){history[group*128+begin+i]=data[2+i];}}}
 counts[group]+=n;if(counts[group]==128){completed+=1;}finish_root();
}
const hp:[*]u32=&history;const cp:[*]u32=&control;const pp:[*]u32=&progress;
comptime{@export_symbol(hp,"history");@export_symbol(cp,"control");@export_symbol(pp,"progress");@export_symbol(initialize);@export_symbol(reset);@export_symbol(run);}
''')
    roots={(16+8*(g%2),g//2):g for g in range(38)}
    lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=64,.height=19});layout{@set_rectangle(64,19);']
    for y in range(19):
        for x in range(64):
            role=-2 if (x,y)==(62,0) else roots.get((x,y),-1)
            lines.append(f'@set_tile_code({x},{y},"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),.source_id=@as(i16,{role})}});')
    lines.append('@export_name("history",[*]u32,false);@export_name("control",[*]u32,false);@export_name("progress",[*]u32,false);@export_name("initialize",fn()void);@export_name("reset",fn()void);@export_name("run",fn()void);}')
    (out/'layout.csl').write_text('\n'.join(lines)+'\n')
    (out/'config.json').write_text(json.dumps(dict(roots=[list(point) for point in roots],cases=[1,2],scope='38 concurrent sources,128 arbitrary u32 words each, five packets each, at FFN gate-root geometry. Two reset-separated calls, exact controller payload and everyPE progress; not numerical MLP or arbitrary bidirectional traffic.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);config=json.loads((root/'config.json').read_text())
    command=['cslc','layout.csl','--arch=wse3','--fabric-dims=71,21','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(command)+'\n');subprocess.run(command,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['history','control','progress']};runner.load();runner.run();reports=[]
    try:
        runner.launch('initialize',nonblock=False)
        for epoch in config['cases']:
            control=np.full(64*19,epoch,np.uint32);runner.memcpy_h2d(ids['control'],control,0,0,64,19,1,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            runner.launch('reset',nonblock=False);runner.launch('run',nonblock=False)
            history=np.zeros(4864,np.uint32);progress=np.zeros(64*19,np.uint32)
            runner.memcpy_d2h(history,ids['history'],62,0,1,1,4864,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            runner.memcpy_d2h(progress,ids['progress'],0,0,64,19,1,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            np.savez(root/f'actual-{epoch}.npz',history=history,progress=progress)
            expected=np.array([0xa5a55a5a^(g*0x01010101)^(i<<12)^(epoch<<24) for g in range(38) for i in range(128)],np.uint32)
            np.testing.assert_array_equal(history,expected)
            state=np.zeros((19,64),np.uint32);state[0,62]=38
            for x,y in config['roots']:state[y,x]=128
            np.testing.assert_array_equal(progress.reshape(19,64),state)
            reports.append(dict(epoch=epoch,actual_sha256=digest(root/f'actual-{epoch}.npz')));print('PACKET FANIN',epoch,flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=reports,scope=config['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

"""SDK message-passing adapter: XY delivery, variable lengths and explicit ACKs."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def digest(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def prepare():
    out=ROOT/'evidence'/('packets-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    for src,dst in [(Path(__file__),'driver.py'),(ROOT/'tools/sdk_probe.py','executor.py'),(ROOT/'csl/runtime/packets.csl','packets.csl')]:shutil.copy2(src,out/dst)
    (out/'pe.csl').write_text('''param memcpy_params;param role:u16;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const packets=@import_module("packets.csl",.{.sent_task_id=@get_local_task_id(12),.received_task_id=@get_local_task_id(13),.on_sent=sent,.on_received=received});
const lengths=[8]u16{1,2,7,30,31,31,2,1};
var control=@zeros([1]u32);var progress=@zeros([1]u32);var history=@constants([248]u32,0xcafebabe);var observed=@zeros([8]u32);var tx=@zeros([31]u32);var sequence:u16=0;var waiting_ack:bool=false;var acked:bool=false;var running:bool=false;var command_waiting:bool=false;
fn initialize() void {packets.init();sys.unblock_cmd_stream();}
fn reset() void {@assert(!packets.sending and !packets.receiving);sequence=0;progress[0]=0;waiting_ack=false;acked=false;running=(role==1);command_waiting=false;const d=@get_dsd(mem1d_dsd,.{.base_address=&history,.extent=248});@mov32(d,0xcafebabe);for(@range(u16,8))|i|{observed[i]=0;}sys.unblock_cmd_stream();}
fn value(seq:u16,i:u16) u32 {return if(i==0) (control[0]<<16)|@as(u32,seq) else @as(u32,0xa5a55a5a)^(@as(u32,i)*0x01010101)^(@as(u32,seq)<<20)^(control[0]<<12);}
fn transmit() void {
 @assert(role==0 and !waiting_ack);const length=lengths[sequence];for(@range(u16,31))|i|{if(i<length){tx[i]=value(sequence,i);}}
 waiting_ack=true;acked=false;packets.send(3,2,&tx,length);
}
fn run() void {if(comptime role==0){running=true;transmit();}else if(comptime role==1){command_waiting=true;if(progress[0]==8){command_waiting=false;sys.unblock_cmd_stream();}}else{sys.unblock_cmd_stream();}}
fn advance() void {
 if(waiting_ack and acked and !packets.sending){waiting_ack=false;sequence+=1;progress[0]=@as(u32,sequence);if(sequence<8){transmit();}else{running=false;sys.unblock_cmd_stream();}}
}
fn sent() void {
 if(comptime role==0){advance();}else if(comptime role==1){progress[0]=@as(u32,sequence);if(sequence==8){running=false;if(command_waiting){command_waiting=false;sys.unblock_cmd_stream();}}}
}
fn received(data:[*]u32,length:u16) void {
 @assert(running);
 if(comptime role==0){
  @assert(waiting_ack and length==3 and data[0]==value(sequence,0) and data[1]==@as(u32,lengths[sequence]));var checksum:u32=0;for(@range(u16,31))|i|{if(i<lengths[sequence]){checksum^=value(sequence,i);}}@assert(data[2]==checksum);acked=true;advance();
 }else if(comptime role==1){
  @assert(sequence<8 and length==lengths[sequence]);var checksum:u32=0;
  for(@range(u16,31))|i|{if(i<length){@assert(data[i]==value(sequence,i));history[sequence*31+i]=data[i];checksum^=data[i];}}
  observed[sequence]=@as(u32,length);tx[0]=data[0];tx[1]=@as(u32,length);tx[2]=checksum;sequence+=1;packets.send(0,0,&tx,3);
 }else{@assert(false);}
}
const cp:[*]u32=&control;const pp:[*]u32=&progress;const hp:[*]u32=&history;const lp:[*]u32=&observed;
comptime{@export_symbol(cp,"control");@export_symbol(pp,"progress");@export_symbol(hp,"history");@export_symbol(lp,"lengths");@export_symbol(initialize);@export_symbol(reset);@export_symbol(run);}
''')
    (out/'layout.csl').write_text('''const memcpy=@import_module("<memcpy/get_params>",.{.width=4,.height=3});layout{@set_rectangle(4,3);for(@range(i16,4))|x|{for(@range(i16,3))|y|{@set_tile_code(x,y,"pe.csl",.{.memcpy_params=memcpy.get_params(x),.role=if(x==0 and y==0) @as(u16,0) else if(x==3 and y==2) @as(u16,1) else @as(u16,2)});}}@export_name("control",[*]u32,false);@export_name("progress",[*]u32,false);@export_name("history",[*]u32,false);@export_name("lengths",[*]u32,false);@export_name("initialize",fn()void);@export_name("reset",fn()void);@export_name("run",fn()void);}
''')
    (out/'config.json').write_text(json.dumps(dict(lengths=[1,2,7,30,31,31,2,1],sessions=[1,2],scope='Public SDK message-passing send with event-driven receive; opposite-corner XY packets and explicit application ACKs, sequential31-wavelet maximum packets. Not concurrent traffic, full model routing or throughput qualification.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);c=json.loads((root/'config.json').read_text());cmd=['cslc','layout.csl','--arch=wse3','--fabric-dims=11,5','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph'];(root/'compile-command.json').write_text(json.dumps(cmd)+'\n');subprocess.run(cmd,check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=8,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ['control','progress','history','lengths']};runner.load();runner.run();reports=[]
    def read(name,x,y,w,h,n):
        data=np.zeros(w*h*n,np.uint32);runner.memcpy_d2h(data,ids[name],x,y,w,h,n,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);return data.reshape(h,w,n)
    try:
        runner.launch('initialize',nonblock=False)
        for session in c['sessions']:
            runner.memcpy_h2d(ids['control'],np.full(12,session,np.uint32),0,0,4,3,1,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);runner.launch('reset',nonblock=False)
            reset=read('progress',0,0,4,3,1);np.save(root/f'reset-{session}.npy',reset);assert np.all(reset==0)
            runner.launch('run',nonblock=False);actual=dict(progress=read('progress',0,0,4,3,1),history=read('history',3,2,1,1,248).reshape(8,31),lengths=read('lengths',3,2,1,1,8).ravel());np.savez(root/f'actual-{session}.npz',**actual)
            expected=np.full((8,31),0xcafebabe,np.uint32)
            for seq,length in enumerate(c['lengths']):
                for i in range(length):expected[seq,i]=(session<<16)|seq if i==0 else 0xa5a55a5a^(i*0x01010101)^(seq<<20)^(session<<12)
            assert np.array_equal(expected,actual['history']) and np.array_equal(actual['lengths'],c['lengths'])
            progress=np.zeros((3,4,1),np.uint32);progress[0,0,0]=progress[2,3,0]=8;assert np.array_equal(progress,actual['progress'])
            reports.append(dict(session=session,actual_sha256=digest(root/f'actual-{session}.npz'),reset_sha256=digest(root/f'reset-{session}.npy')));print('PACKETS SESSION',session,flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=reports,scope=c['scope']),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),1800)

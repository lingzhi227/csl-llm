"""Odd/even source columns, changing row strides and continuous BF16 expansion."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def prepare(negative):
    out=ROOT/'evidence'/('bf16-stride-'+('negative-' if negative else '')+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    shutil.copy2(__file__,out/'driver.py');shutil.copy2(ROOT/'tools/sdk_probe.py',out/'executor.py');text=(ROOT/'csl/kernels/local_gemv_bf16_f32.csl').read_text()
    if negative:text=text.replace('const base = @get_dsd(mem1d_dsd, .{ .base_address = weights, .extent = rows, .stride = columns });\n  const wd = @increment_dsd_offset(base, @as(i16, column), u16);','const wd = @get_dsd(mem1d_dsd, .{ .base_address = &weights[column], .extent = rows, .stride = columns });')
    (out/'kernel.csl').write_text(text)
    (out/'config.json').write_text(json.dumps(dict(negative_old_base=negative))+'\n')
    (out/'layout.csl').write_text('''const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("words",[*]u16,false);@export_name("control",[*]u16,false);@export_name("observed",[*]u32,false);@export_name("progress",[*]u32,false);@export_name("initialize",fn()void);@export_name("expand",fn()void);}
''')
    (out/'pe.csl').write_text('''param memcpy_params;const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const wide=@import_module("kernel.csl",.{.rows=64,.columns=224});const narrow=@import_module("kernel.csl",.{.rows=64,.columns=3});
var words=@zeros([14336]u16);var control=@zeros([1]u16);var observed=@zeros([1024]u32);var progress=@zeros([1]u32);
const wp:[*]u16=&words;const cp:[*]u16=&control;const op:[*]u32=&observed;const pp:[*]u32=&progress;
fn initialize() void {
 const a:[*]u32=@ptrcast([*]u32,&wide.expanded);const b:[*]u32=@ptrcast([*]u32,&narrow.expanded);
 for(@range(u16,64))|i|{a[i]=0xa5a55a5a;b[i]=0xa5a55a5a;}
 wide.reset_expansion();narrow.reset_expansion();progress[0]=0;sys.unblock_cmd_stream();
}
fn expand() void {
 for(@range(u16,16))|i|{
  if(control[0]<14){
   const base=control[0]*8+i/2;const column=if(i%2==0)base else 223-base;
   wide.expand_column(&words,column);
   for(@range(u16,64))|row|{observed[i*64+row]=@bitcast(u32,wide.expanded[row]);}
  }else{
   narrow.expand_column(&words,(i+control[0])%3);
   for(@range(u16,64))|row|{observed[i*64+row]=@bitcast(u32,narrow.expanded[row]);}
  }
 }
 progress[0]+=1;sys.unblock_cmd_stream();
}
comptime{@export_symbol(wp,"words");@export_symbol(cp,"control");@export_symbol(op,"observed");@export_symbol(pp,"progress");@export_symbol(initialize);@export_symbol(expand);}
''')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify,sha
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);subprocess.run(['cslc','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1'],check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=4,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ('words','control','observed','progress')};runner.load();runner.run();rows=[]
    words=(np.arange(14336,dtype=np.uint32)*2371+4099)&65535;np.save(root/'inputs.npy',words)
    try:
        runner.memcpy_h2d(ids['words'],words,0,0,1,1,len(words),streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);runner.launch('initialize',nonblock=False)
        for epoch in range(16):
            runner.memcpy_h2d(ids['control'],np.array([epoch],np.uint32),0,0,1,1,1,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);runner.launch('expand',nonblock=False)
            actual=np.zeros(1024,np.uint32);progress=np.zeros(1,np.uint32)
            for name,data in [('observed',actual),('progress',progress)]:runner.memcpy_d2h(data,ids[name],0,0,1,1,len(data),streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            np.savez(root/f'actual-{epoch}.npz',words=actual,progress=progress)
            columns=[epoch*8+i//2 if i%2==0 else 223-(epoch*8+i//2) for i in range(16)] if epoch<14 else [(i+epoch)%3 for i in range(16)]
            stride=224 if epoch<14 else 3;expected=np.stack([words[np.arange(64)*stride+column]<<16 for column in columns]).ravel();np.testing.assert_array_equal(actual,expected);assert progress[0]==epoch+1
            rows.append(dict(epoch=epoch,columns=columns,stride=stride,progress=progress.tolist(),actual_sha256=sha(root/f'actual-{epoch}.npz')));print('STRIDE CALL',epoch+1,flush=True)
    finally:runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=rows,inputs_sha256=sha(root/'inputs.npy'),no_reset_between_columns=True,scope='Exact shared production expansion: every column0..223, alternating odd/even and increasing/decreasing addresses; stride224 and3; continuous loads after poisoned-buffer reset.'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--negative-old-base',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare(a.negative_old_base)
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),900)

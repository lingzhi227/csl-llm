"""SDK raw-word validation of the exact shared GEMV BF16 expansion function."""
import argparse,datetime,hashlib,json,os,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def prepare():
    out=ROOT/'evidence'/('bf16-bits-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
    shutil.copy2(__file__,out/'driver.py');shutil.copy2(ROOT/'tools/sdk_probe.py',out/'executor.py');shutil.copy2(ROOT/'csl/kernels/local_gemv_bf16_f32.csl',out/'kernel.csl')
    (out/'layout.csl').write_text('''const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});
layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("words",[*]u16,false);@export_name("expanded",[*]f32,false);@export_name("progress",[*]u32,false);@export_name("expand",fn()void);}
''')
    (out/'pe.csl').write_text('''param memcpy_params;const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const kernel=@import_module("kernel.csl",.{.rows=2048,.columns=1});
var words=@zeros([2048]u16);var progress=@zeros([1]u32);
const input_ptr:[*]u16=&words;const raw_ptr:[*]f32=&kernel.expanded;const progress_ptr:[*]u32=&progress;
fn expand() void {
 const poison:[*]u32=@ptrcast([*]u32,&kernel.expanded);
 for(@range(u16,2048))|i|{poison[i]=0xa5a55a5a;}
 kernel.reset_expansion();kernel.expand_column(&words,0);progress[0]+=1;sys.unblock_cmd_stream();
}
comptime{@export_symbol(input_ptr,"words");@export_symbol(raw_ptr,"expanded");@export_symbol(progress_ptr,"progress");@export_symbol(expand);}
''')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}),indent=2)+'\n');print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify,sha
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
    verify(root);os.chdir(root);subprocess.run(['cslc','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1'],check=True)
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=4,dump_core=True),SdkTarget.WSE3));ids={n:runner.get_id(n) for n in ('words','expanded','progress')};runner.load();runner.run();rows=[];all_actual=[];all_inputs=[]
    try:
        for epoch in range(64):
            inputs=np.arange((epoch%32)*2048,(epoch%32+1)*2048,dtype=np.uint32)
            if epoch>=32:inputs^=65535
            runner.memcpy_h2d(ids['words'],inputs,0,0,1,1,2048,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False);runner.launch('expand',nonblock=False)
            actual=np.zeros(2048,np.uint32);progress=np.zeros(1,np.uint32)
            for name,data in [('expanded',actual),('progress',progress)]:runner.memcpy_d2h(data,ids[name],0,0,1,1,len(data),streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
            all_actual.append(actual.copy());all_inputs.append(inputs.copy());np.testing.assert_array_equal(actual,inputs<<16);assert progress[0]==epoch+1;rows.append(dict(epoch=epoch,progress_raw=progress.tolist(),input_words=int(inputs.size),output_words=int(actual.size),input_sha256=hashlib.sha256(inputs.tobytes()).hexdigest(),output_sha256=hashlib.sha256(actual.tobytes()).hexdigest(),bitwise_equal=True,low_halfword_zero=bool(np.all((actual&65535)==0))))
            print('BIT CALL',epoch+1,flush=True)
    finally:
        np.save(root/'inputs.npy',np.stack(all_inputs) if all_inputs else np.zeros((0,2048),np.uint32));np.save(root/'actual.npy',np.stack(all_actual) if all_actual else np.zeros((0,2048),np.uint32));runner.stop()
    (root/'results.json').write_text(json.dumps(dict(success=True,calls=rows,unique_bf16_patterns=65536,checked_words=131072,poison_reset_each_call=True,actual_sha256=sha(root/'actual.npy'),inputs_sha256=sha(root/'inputs.npy'),scope='Raw expansion bits, including signed zero/subnormals/infinity/NaN payloads; not floating arithmetic on nonfinite values.'),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--prepare',action='store_true');p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker:worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()));from executor import execute
        execute(a.execute.resolve(),900)

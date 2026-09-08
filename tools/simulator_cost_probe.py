"""Same-binary no-op/GEMV/no-op experiment with explicit SDK task waits.

Wall timings name API boundaries; they are not hardware latency measurements.
Both paths retain identical layout, ELF memory, initial weights and readbacks.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def prepare(width, height, threads):
    assert 1 <= width <= 192 and 1 <= height <= 210
    assert 1 <= threads <= 16
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out = ROOT / 'evidence' / f'simulator-cost-{width}x{height}-t{threads}-{stamp}'
    out.mkdir()
    for source, name in [(Path(__file__), 'driver.py'),
                         (ROOT / 'tools/sdk_probe.py', 'executor.py'),
                         (ROOT / 'csl/kernels/local_gemv_bf16_f32_colmajor.csl', 'kernel.csl')]:
        shutil.copy2(source, out / name)
    (out / 'pe.csl').write_text('''param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const clock=@import_module("<time>");
const kernel=@import_module("kernel.csl",.{.rows=128,.columns=112});
var weights=@constants([14336]u16,0x3c80);
var input=@constants([112]f32,0.125);var output=@zeros([128]f32);
var mode=@zeros([1]u32);var progress=@zeros([1]u32);var timing=@zeros([6]u16);
var started=@zeros([3]u16);var finished=@zeros([3]u16);
fn compute() void {
 clock.enable_tsc();clock.get_timestamp(&started);
 if(mode[0]==1){kernel.compute(&weights,&input,&output);}
 else{const d=@get_dsd(mem1d_dsd,.{.base_address=&output,.extent=128});@fmovs(d,0.21875);}
 clock.get_timestamp(&finished);clock.disable_tsc();
 for(@range(u16,3))|i|{timing[i]=started[i];timing[i+3]=finished[i];}
 progress[0]+=1;sys.unblock_cmd_stream();
}
const op:[*]f32=&output;const mp:[*]u32=&mode;const pp:[*]u32=&progress;const tp:[*]u16=&timing;
comptime{@export_symbol(op,"output");@export_symbol(mp,"mode");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(compute);}
''')
    (out / 'layout.csl').write_text(f'''const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height}}});
layout{{@set_rectangle({width},{height});
 for(@range(i16,{width}))|x|{{for(@range(i16,{height}))|y|{{@set_tile_code(x,y,"pe.csl",.{{.memcpy_params=memcpy.get_params(x)}});}}}}
 @export_name("output",[*]f32,false);@export_name("mode",[*]u32,false);@export_name("progress",[*]u32,false);@export_name("timing",[*]u16,false);@export_name("compute",fn()void);
}}
''')
    (out / 'config.json').write_text(json.dumps(dict(width=width, height=height, threads=threads,
        modes=[0, 1, 0], points=sorted(set([(0, 0), (width-1, 0), (0, height-1), (width-1, height-1)])),
        scope='Same compiled binary and readback regions. No-op fills output; GEMV computes it. Sampled correctness only. API wall costs may include simulator scheduling and transport.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d', files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify, sha
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    verify(root)
    os.chdir(root)
    c = json.loads((root / 'config.json').read_text())
    w, h = c['width'], c['height']
    events = []
    def timed(name, operation):
        begin = time.monotonic()
        (root/'stage.json').write_text(json.dumps(dict(stage=name, time=time.time()))+'\n')
        value = operation()
        events.append(dict(name=name, seconds=time.monotonic()-begin))
        (root/'wall-events.json').write_text(json.dumps(events, indent=2)+'\n')
        print(name, events[-1]['seconds'], flush=True)
        return value
    cmd = ['cslc','layout.csl','--arch=wse3',f'--fabric-dims={w+7},{h+2}', '--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--dump-dsr-alloc-graph']
    (root/'compile-command.json').write_text(json.dumps(cmd)+'\n')
    timed('compile', lambda: subprocess.run(cmd, check=True))
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(num_threads=c['threads'], suppress_trace=True, dump_core=True), SdkTarget.WSE3))
    ids = {n:runner.get_id(n) for n in ('mode','output','progress','timing')}
    timed('load', runner.load)
    timed('run', runner.run)
    cases = []
    try:
        for epoch, mode in enumerate(c['modes']):
            timed(f'{epoch}-set-mode', lambda: runner.memcpy_h2d(ids['mode'],np.full(w*h,mode,np.uint32),0,0,w,h,1,streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False))
            task = timed(f'{epoch}-launch-nonblocking', lambda: runner.launch('compute', nonblock=True))
            timed(f'{epoch}-explicit-task-wait', lambda: runner.task_wait(task))
            raw = {}
            for name, n in [('output',128),('progress',1),('timing',6)]:
                parts = []
                for x, y in c['points']:
                    data = np.zeros(n, np.float32 if name=='output' else np.uint32)
                    timed(f'{epoch}-read-{name}-{x}-{y}', lambda: runner.memcpy_d2h(data,ids[name],x,y,1,1,n,streaming=False,data_type=MemcpyDataType.MEMCPY_16BIT if name=='timing' else MemcpyDataType.MEMCPY_32BIT,order=MemcpyOrder.ROW_MAJOR,nonblock=False))
                    parts.append(data)
                raw[name] = np.stack(parts)
            np.savez(root/f'actual-{epoch}.npz', **raw)
            assert np.all(raw['output']==np.float32(.21875))
            assert np.all(raw['progress']==epoch+1)
            t = raw['timing'].astype(np.uint64)
            assert np.all(t<65536)
            ticks = lambda v:v[:,0]+(v[:,1]<<16)+(v[:,2]<<32)
            cycles = (ticks(t[:,3:])-ticks(t[:,:3])) & np.uint64((1<<48)-1)
            assert np.all((cycles>0)&(cycles<1<<40))
            cases.append(dict(epoch=epoch,mode=mode,cycles=cycles.tolist(),actual_sha256=sha(root/f'actual-{epoch}.npz')))
            (root/'results.json').write_text(json.dumps(dict(success=False,cases=cases),indent=2)+'\n')
    finally:
        timed('stop',runner.stop)
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=cases,wall_events_sha256=sha(root/'wall-events.json'),scope=c['scope']),indent=2)+'\n')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--prepare',action='store_true');p.add_argument('--width',type=int,default=1);p.add_argument('--height',type=int,default=1);p.add_argument('--threads',type=int,default=8);p.add_argument('--worker',type=Path);p.add_argument('--execute',type=Path)
    a=p.parse_args()
    if a.prepare: prepare(a.width,a.height,a.threads)
    elif a.worker: worker(a.worker.resolve())
    else:
        sys.path.insert(0,str(a.execute.resolve()))
        from executor import execute
        execute(a.execute.resolve(),1800)

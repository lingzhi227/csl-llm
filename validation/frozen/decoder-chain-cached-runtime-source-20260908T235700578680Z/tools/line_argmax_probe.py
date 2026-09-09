"""Freeze a sparse four-participant exact argmax protocol probe.

No model arithmetic or claimed head speedup. Do not overlap an active large SDK
job; the frozen executor bounds this standalone future probe to300s/2GiB.
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

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(source, project):
    import numpy as np
    sys.path.insert(0, str(source / 'src'))
    from csl_llm.argmax_routes import emit
    out = project / 'evidence' / ('line-argmax-'+datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    values = np.array([[1, 4, -2, 3], [5, 5, 5, 5], [-0., 0., -0., 0.],
                       [-7, -1, -1, -4], [1, 4, -2, 3]], np.float32)
    indices = np.array([[100, 200, 300, 400], [151935, 7, 128, 0], [0, 3, 2, 1],
                        [151935, 100, 2, 0], [100, 200, 300, 400]], np.uint32)
    np.savez(out / 'inputs.npz', values=values, indices=indices,
             epochs=np.array([1, 2, 3, 65535, 1], np.uint32))
    lines = ['const memcpy=@import_module("<memcpy/get_params>",.{.width=10,.height=1});',
             'layout{@set_rectangle(10,1);']
    for x in range(10):
        lines.append(f'@set_tile_code({x},0,"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),'
            f'.participant={str(x%3==0).lower()},.ordinal={x//3 if x%3==0 else 0}}});')
    lines.append(emit(4, 3))
    for name, kind in [('data','u32'), ('control','u32'), ('status','u32'), ('timing','u16')]:
        lines.append(f'@export_name("{name}",[*]{kind},false);')
    for name in ['initialize','prepare','compute']:
        lines.append(f'@export_name("{name}",fn()void);')
    (out / 'layout.csl').write_text('\n'.join(lines+['}'])+'\n')
    for path, name in [(Path(__file__),'driver.py'), (source/'tools/sdk_probe.py','executor.py'),
                      (source/'csl/programs/line_argmax/pe.csl','pe.csl'),
                      (source/'csl/runtime/line_argmax_f32.csl','line_argmax.csl'),
                      (source/'csl/kernels/argmax_f32.csl','argmax.csl'),
                      (source/'src/csl_llm/argmax_routes.py','argmax_routes.py')]:
        shutil.copy2(path,out/name)
    (out/'config.json').write_text(json.dumps(dict(width=10,height=1,participants=4,stride=3,
        cases=5,timeout_seconds=300,rss_limit_kib=2*1024*1024,
        scope='Four sparse participants, five globally drained frames: exact finite FP32/uint32 '
              'suffix argmax, nonmonotone tie IDs, signed zero, negatives and repeat. '
              'Epoch65535-to1 is this probe only, not full-model reset/wrap. '
              'No original vocabulary arithmetic, full-model integration or measured head speedup.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(
        sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    verify(root);os.chdir(root)
    command=['cslc','layout.csl','--arch=wse3','--fabric-dims=17,3','--fabric-offsets=4,1',
             '-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']
    (root/'compile-command.json').write_text(json.dumps(command)+'\n')
    subprocess.run(command,check=True)
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform)
    data=np.load(root/'inputs.npz')
    runner=SdkRuntime('out',get_platform(None,SimfabConfig(
        suppress_trace=True,num_threads=1,dump_core=True),SdkTarget.WSE3))
    ids={name:runner.get_id(name) for name in ['data','control','status','timing']}
    options=dict(streaming=False,data_type=MemcpyDataType.MEMCPY_32BIT,
                 order=MemcpyOrder.ROW_MAJOR,nonblock=False)
    records=[]
    try:
        runner.load();runner.run();runner.launch('initialize',nonblock=False)
        for case,epoch in enumerate(data['epochs']):
            payload=np.zeros((10,2),np.uint32)
            payload[::3,0]=data['values'][case].view(np.uint32)
            payload[::3,1]=data['indices'][case]
            runner.memcpy_h2d(ids['data'],payload.ravel(),0,0,10,1,2,**options)
            runner.memcpy_h2d(ids['control'],np.full(10,epoch,np.uint32),0,0,10,1,1,**options)
            runner.launch('prepare',nonblock=False)
            runner.launch('compute',nonblock=False)
            status=np.zeros((10,4),np.uint32)
            runner.memcpy_d2h(status.ravel(),ids['status'],0,0,10,1,4,**options)
            # Inspect every participant and forwarder before the next epoch.
            np.testing.assert_array_equal(status[:,0],np.full(10,epoch,np.uint32))
            np.testing.assert_array_equal(status[:,1],np.ones(10,np.uint32))
            for ordinal in range(4):
                candidates=range(ordinal,4)
                winner=min(candidates,key=lambda i:(-float(data['values'][case,i]),int(data['indices'][case,i])))
                assert status[ordinal*3,2]==data['values'][case].view(np.uint32)[winner]
                assert status[ordinal*3,3]==data['indices'][case,winner]
            np.testing.assert_array_equal(status[[1,2,4,5,7,8],2:],np.zeros((6,2),np.uint32))
            timing=np.zeros(6,np.uint32)
            runner.memcpy_d2h(timing,ids['timing'],0,0,1,1,6,
                             **dict(options,data_type=MemcpyDataType.MEMCPY_16BIT))
            ticks=[sum(int(timing[k+i])<<(16*i) for i in range(3)) for k in [0,3]]
            cycles=(ticks[1]-ticks[0])&((1<<48)-1)
            assert 0<cycles<1<<40
            actual=root/f'actual-{case}.npz'
            np.savez(actual,status=status,timing=timing)
            records.append(dict(case=case,epoch=int(epoch),root_cycles=cycles,
                                exact=True,actual_sha256=digest(actual)))
    finally:
        runner.stop()
    np.testing.assert_array_equal(np.load(root/'actual-0.npz')['status'],
                                  np.load(root/'actual-4.npz')['status'])
    (root/'results.json').write_text(json.dumps(dict(success=True,cases=records,
        scope=json.loads((root/'config.json').read_text())['scope']),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare',action='store_true')
    action.add_argument('--worker',type=Path)
    action.add_argument('--execute',type=Path)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.prepare:prepare(args.source_root.resolve(),args.project_root.resolve())
    elif args.worker:worker(args.worker.resolve())
    else:
        sys.path.insert(0,str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(),300,rss_limit_kib=2*1024*1024)

"""Self-contained read-only audit of contracted addresses in an actual core.

No SdkRuntime or compiler is constructed. Report/ELF provenance is verified in
preparation, frozen, and checked again through its complete receipt chain.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def prepare(runtime,case,source,project):
    from sdk_probe import verify
    verify(runtime)
    directory=runtime/f'case-{case}'
    result=json.loads((directory/'results.json').read_text())
    assert result['success'] and result['case']==case
    assert result['runtime_manifest_sha256']==digest(runtime/'manifest.json')
    for name,key in [('out.core','core_sha256'),('actual.npz','actual_sha256'),
                     ('normal-d2h.npz','normal_d2h_sha256'),('runtime-artifacts.json','runtime_artifacts_sha256')]:
        assert digest(directory/name)==result[key]
    compiled=json.loads((runtime/'compiled-files.json').read_text())
    assert digest(runtime/'compiled-files.json')==result['compiled_files_sha256']
    for name,expected in compiled.items():
        assert digest(directory/'out'/name)==expected
    artifacts=json.loads((directory/'runtime-artifacts.json').read_text())
    for name,expected in artifacts.items():
        assert digest(directory/name)==expected
    out=project/'evidence'/('core-contract-audit-'+datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    copied=[(runtime/'manifest.json','source-runtime-manifest.json'),
            (directory/'results.json','source-case-results.json'),
            (runtime/'compiled-files.json','compiled-files.json'),
            (directory/'runtime-artifacts.json','runtime-artifacts.json'),
            (runtime/'source-symbols.json','source-symbols.json'),
            (runtime/'source-composition.json','source-composition.json'),
            (runtime/'weights.u32.bin','weights.u32.bin'),
            (directory/'out.core','actual.core'),(directory/'actual.npz','expected.npz'),
            (directory/'normal-d2h.npz','normal-d2h.npz'),
            (Path(__file__),'driver.py'),(source/'tools/sdk_probe.py','executor.py'),
            (source/'tools/sdk_core.py','sdk_core.py')]
    for path,name in copied:shutil.copy2(path,out/name)
    (out/'config.json').write_text(json.dumps(dict(source_run=runtime.name,case=case,
        source_case_results_sha256=digest(directory/'results.json'),timeout_seconds=180,rss_limit_kib=2*1024*1024,
        scope='Read-only inspection of an already executed full vocabulary core using previously '
              'audited hash-bound symbol addresses. Full original weights and actual arrays/normal '
              'D2H are compared. No new simulator/compiler/neural result or hardware performance. '
              'Timing is not isolated from other host work.'),indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(
        sdk_sha256=json.loads((runtime/'manifest.json').read_text())['sdk_sha256'],
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    print(out,flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from sdk_core import StoppedCore
    started=time.monotonic();verify(root);os.chdir(root)
    config=json.loads((root/'config.json').read_text())
    previous=json.loads((root/'source-case-results.json').read_text())
    assert digest(root/'source-case-results.json')==config['source_case_results_sha256']
    assert previous['success'] and previous['case']==config['case']
    assert digest(root/'source-runtime-manifest.json')==previous['runtime_manifest_sha256']
    runtime_manifest=json.loads((root/'source-runtime-manifest.json').read_text())
    assert runtime_manifest['files']['inputs.npz']=='ba7327dfbe631a83dbf8655110f3fb43938d58f71cfd9043122fee6f0d30d499'
    for name,key in [('source-symbols.json','source_symbols_sha256'),
                     ('source-composition.json','source_composition_sha256'),
                     ('compiled-files.json','compiled_files_sha256')]:
        assert digest(root/name)==previous[key]==runtime_manifest['files'][name]
    for name,key in [('actual.core','core_sha256'),('expected.npz','actual_sha256'),
                     ('normal-d2h.npz','normal_d2h_sha256'),('runtime-artifacts.json','runtime_artifacts_sha256')]:
        assert digest(root/name)==previous[key]
    assert digest(root/'weights.u32.bin')==previous['original_weights_sha256']==runtime_manifest['files']['weights.u32.bin']
    compiled=json.loads((root/'compiled-files.json').read_text())
    artifacts=json.loads((root/'runtime-artifacts.json').read_text())
    contract=json.loads((root/'source-symbols.json').read_text())
    composition=json.loads((root/'source-composition.json').read_text())
    assert compiled==composition['artifact_sha256']
    assert all(artifacts['out/'+name]==expected for name,expected in compiled.items())
    proven=time.monotonic()
    core=StoppedCore.from_verified_contract(root/'actual.core',contract,composition)
    loaded=time.monotonic()
    expected=np.load(root/'expected.npz');normal=np.load(root/'normal-d2h.npz')
    def read(x,y,name,count,dtype='<u4'):
        dt=np.dtype(dtype)
        return np.frombuffer(core.read(x+4,y+1,name,count*dt.itemsize),dtype=dt).copy()
    progress=np.empty((50,193),np.uint32)
    with (root/'weights.u32.bin').open('rb') as packed:
        for y in range(50):
            for x in range(193):
                matrix=x<192 and y*192+x<9496
                progress[y,x]=read(x,y,'progress',1)[0]
                assert int(read(x,y,'identity',1)[0])==((y*192+x)//8 if matrix else 0)
                token=[0,151645,151935][config['case']]
                assert read(x,y,'control',2).tolist()==[1,token]
                if matrix:
                    raw=packed.read(28672);assert len(raw)==28672
                    assert core.read(x+4,y+1,'weights',28672)==raw
        assert not packed.read(1)
    np.testing.assert_array_equal(progress,expected['progress']);assert np.all(progress==4)
    logits=np.concatenate([read((group%24)*8,group//24,'result',128,'<f4') for group in range(1187)])
    lookup=read(192,0,'lookup',896,'<f4')
    timing=read(192,0,'timing',6,'<u2')
    for actual,name in [(logits,'logits'),(lookup,'lookup')]:
        assert actual.tobytes()==expected[name].tobytes()
    np.testing.assert_array_equal(timing,expected['timing'])
    for name in ['winner','best']:
        assert core.read(196,1,name,4)==normal[name].tobytes()
    completed=time.monotonic()
    (root/'results.json').write_text(json.dumps(dict(success=True,offline_only=True,
        case=config['case'],matrix_tiles=9496,application_coordinates=9650,logits=151936,lookup=896,
        all_original_weights_exact=True,all_actual_arrays_exact=True,
        source_case_results_sha256=config['source_case_results_sha256'],
        core_sha256=digest(root/'actual.core'),
        proof_checks_and_json_seconds=proven-started,contract_core_load_seconds=loaded-proven,
        read_compare_seconds=completed-loaded,verified_read_seconds=completed-started,
        scope=config['scope']),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    action=parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-runtime',type=Path)
    action.add_argument('--worker',type=Path)
    action.add_argument('--execute',type=Path)
    parser.add_argument('--case',type=int,choices=range(3),default=0)
    parser.add_argument('--source-root',type=Path,default=ROOT)
    parser.add_argument('--project-root',type=Path,default=ROOT)
    args=parser.parse_args()
    if args.from_runtime:
        prepare(args.from_runtime.resolve(),args.case,args.source_root.resolve(),args.project_root.resolve())
    elif args.worker:worker(args.worker.resolve())
    else:
        sys.path.insert(0,str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(),180,rss_limit_kib=2*1024*1024)

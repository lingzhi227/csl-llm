"""Bounded host-only construction of the independent two-token reference.

This does not invoke the SDK, compiler or simulator. Reuse the existing process
identity/cleanup helpers while retaining a separate CPU-only execution receipt.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]


def prepare(project, source):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import sha
    python = project / '.venv/bin/python'
    if not python.is_file():
        raise ValueError('Pinned CPU reference environment is required')
    out = project / 'evidence' / ('decoder-chain-reference-build-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    names = ['tools/decoder_chain_reference.py', 'tools/resident_decoder_probe.py',
             'tools/sdk_probe.py', 'src/csl_llm/__init__.py', 'src/csl_llm/decoder_layout.py',
             'src/csl_llm/regions.py', 'src/csl_llm/static_data.py', 'configs/precision.json']
    for name in names:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'process_helpers.py')
    (out / 'config.json').write_text(json.dumps(dict(project_root=str(project),
        python=str(python), python_sha256=sha(python), timeout_seconds=180,
        rss_limit_kib=2*1024*1024, threads=1, prefix_length=2,
        scope='Host CPU independent two-token reference construction only; no SDK execution.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(out)): sha(p)
        for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path.insert(0, str(root / 'source/tools'))
    from sdk_probe import verify, sha
    from decoder_chain_reference import prepare as build
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    assert config['prefix_length'] == 2 and config['threads'] == 1
    reference = build(Path(config['project_root']), root / 'source', 2)
    (root / 'results.json').write_text(json.dumps(dict(success=True, cpu_only=True,
        reference_bundle=str(reference), reference_manifest_sha256=sha(reference / 'manifest.json'),
        reference_inputs_sha256=sha(reference / 'inputs.npz'), scope=config['scope']), indent=2)+'\n')


def execute(root):
    sys.path.insert(0, str(root))
    from process_helpers import verify, sha, process_identity, finalize_tree
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    assert (config['timeout_seconds'], config['rss_limit_kib'], config['threads']) == (180, 2*1024*1024, 1)
    assert sha(Path(config['python'])) == config['python_sha256']
    if (root / 'cpu.log').exists():
        raise ValueError('An existing CPU run must not be restarted')
    command = [config['python'], '-u', str(root / 'driver.py'), '--worker', str(root)]
    receipt = dict(success=False, cpu_only=True, command=command,
        manifest_sha256=sha(root / 'manifest.json'), python_sha256=config['python_sha256'],
        timeout_seconds=180, rss_limit_kib=2*1024*1024, peak_rss_kib=0,
        rss_kind='Sum of process-tree resident pages; shared mappings may be counted twice')
    start, known, samples = time.monotonic(), {}, []
    with (root / 'cpu.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
            start_new_session=True, env=dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
                OPENBLAS_NUM_THREADS='1', NUMEXPR_NUM_THREADS='1', PYTHONUNBUFFERED='1'))
        try:
            while process.poll() is None:
                rows = [tuple(map(int, line.split())) for line in subprocess.check_output(
                    ['ps', '-eo', 'pid=,ppid=,rss='], text=True).splitlines() if line.strip()]
                family, previous = {process.pid}, None
                while previous != family:
                    previous = set(family)
                    family.update(pid for pid, parent, _ in rows if parent in family)
                known.update({pid: process_identity(pid) for pid in family if pid not in known})
                rss = sum(rss for pid, _, rss in rows if pid in family)
                elapsed = time.monotonic()-start
                receipt['peak_rss_kib'] = max(receipt['peak_rss_kib'], rss)
                samples.append(dict(elapsed_seconds=elapsed, rss_kib=rss, processes=len(family)))
                if elapsed > 180 or rss > 2*1024*1024:
                    raise TimeoutError('CPU reference resource/time budget reached')
                time.sleep(0.25)
            assert process.returncode == 0, process.returncode
            result = json.loads((root / 'results.json').read_text())
            assert result['success'] and result['cpu_only']
            receipt.update(success=True, results_sha256=sha(root / 'results.json'))
        except BaseException:
            receipt['error'] = traceback.format_exc()
            raise
        finally:
            receipt['observed_process_identities'] = {str(pid): value for pid, value in known.items()}
            receipt['cleanup_signals'], receipt['after_cleanup_identities'] = finalize_tree(process, known)
            receipt['elapsed_seconds'] = time.monotonic()-start
            (root / 'resource-samples.json').write_text(json.dumps(samples)+'\n')
            receipt['samples_sha256'] = sha(root / 'resource-samples.json')
            (root / 'execution.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print('CPU REFERENCE BUILD PASS', root, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.project_root.resolve(), args.source_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        execute(args.execute.resolve())

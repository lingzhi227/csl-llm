"""Bounded host-only construction of explicit independent reference scopes.

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


def budget(kind):
    if kind == 'decoder-chain':
        return 180, 2*1024*1024
    if kind == 'model-single-token':
        return 180, 4*1024*1024
    raise ValueError('Unknown CPU reference scope')


def prepare(project, source, *, kind='decoder-chain'):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import sha
    python = project / '.venv/bin/python'
    if not python.is_file():
        raise ValueError('Pinned CPU reference environment is required')
    seconds, memory = budget(kind)
    label = 'decoder-chain-reference-build-' if kind == 'decoder-chain' else 'model-reference-build-'
    out = project / 'evidence' / (label + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    names = ['tools/decoder_chain_reference.py', 'tools/resident_decoder_probe.py',
             'tools/sdk_probe.py', 'src/csl_llm/__init__.py', 'src/csl_llm/decoder_layout.py',
             'src/csl_llm/regions.py', 'src/csl_llm/static_data.py', 'configs/precision.json']
    if kind == 'model-single-token':
        names.append('tools/model_single_token_reference.py')
    for name in names:
        destination = out / 'source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'process_helpers.py')
    (out / 'config.json').write_text(json.dumps(dict(project_root=str(project),
        python=str(python), python_sha256=sha(python), timeout_seconds=seconds,
        rss_limit_kib=memory, threads=1, reference_kind=kind,
        prefix_length=2 if kind == 'decoder-chain' else 1,
        scope='Host CPU reference construction only: '+kind+'; no SDK execution.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(files={str(p.relative_to(out)): sha(p)
        for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path.insert(0, str(root / 'source/tools'))
    from sdk_probe import verify, sha
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    kind = config.get('reference_kind', 'decoder-chain')
    assert config['threads'] == 1
    extra = {}
    if kind == 'decoder-chain':
        from decoder_chain_reference import prepare as build
        assert config['prefix_length'] == 2
        reference = build(Path(config['project_root']), root / 'source', 2)
        extra['reference_inputs_sha256'] = sha(reference / 'inputs.npz')
    elif kind == 'model-single-token':
        from model_single_token_reference import prepare as build
        assert config['prefix_length'] == 1
        reference = build(Path(config['project_root']), root / 'source')
        extra['reference_report_sha256'] = sha(reference / 'report.json')
        extra['reference_trace_sha256'] = sha(reference / 'single-token/step-000.npz')
    else:
        raise ValueError('Unknown CPU reference worker')
    (root / 'results.json').write_text(json.dumps(dict(success=True, cpu_only=True,
        reference_bundle=str(reference), reference_manifest_sha256=sha(reference / 'manifest.json'),
        **extra, scope=config['scope']), indent=2)+'\n')


def execute(root):
    sys.path.insert(0, str(root))
    from process_helpers import verify, sha, process_identity, finalize_tree
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    seconds, memory = budget(config.get('reference_kind', 'decoder-chain'))
    assert (config['timeout_seconds'], config['rss_limit_kib'], config['threads']) == (seconds, memory, 1)
    assert sha(Path(config['python'])) == config['python_sha256']
    if (root / 'cpu.log').exists():
        raise ValueError('An existing CPU run must not be restarted')
    command = [config['python'], '-u', str(root / 'driver.py'), '--worker', str(root)]
    receipt = dict(success=False, cpu_only=True, command=command,
        manifest_sha256=sha(root / 'manifest.json'), python_sha256=config['python_sha256'],
        timeout_seconds=seconds, rss_limit_kib=memory, peak_rss_kib=0,
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
                if elapsed > seconds or rss > memory:
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
    parser.add_argument('--kind', choices=['decoder-chain', 'model-single-token'], default='decoder-chain')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.project_root.resolve(), args.source_root.resolve(), kind=args.kind)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        execute(args.execute.resolve())

"""Self-contained bounded compatibility check of two completed model audits."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(candidate, baseline, source, project):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import verify, sha
    out = project / 'evidence' / ('model-contract-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    # Verify completion before making the new immutable input snapshot.
    executions = {}
    for name, root in [('candidate', candidate), ('baseline', baseline)]:
        verify(root)
        execution = json.loads((root / 'execution.json').read_text())
        result = json.loads((root / 'results.json').read_text())
        assert execution['success'] and result['success'] and result['offline_only']
        assert execution['manifest_sha256'] == sha(root / 'manifest.json')
        assert execution['results_sha256'] == sha(root / 'results.json')
        executions[name] = sha(root / 'execution.json')
    out.mkdir()
    for name, root in [('candidate', candidate), ('baseline', baseline)]:
        shutil.copytree(root, out / name, ignore=shutil.ignore_patterns('__pycache__'))
    names = ['tools/model_partition_contract.py', 'tools/sdk_probe.py',
             'tools/elf_partition.py', 'tools/compare_elf_images.py',
             'src/csl_llm/__init__.py', 'src/csl_llm/decoder_layout.py', 'src/csl_llm/regions.py']
    for name in names:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    (out / 'config.json').write_text(json.dumps(dict(execution_sha256=executions,
        timeout_seconds=180, rss_limit_kib=2*1024*1024,
        scope='Offline whole-ELF partition containment, common allocated images, fixed CSL/layout '
              'and complete output metadata/ordered SDK I/O/RPC compatibility. No composition, '
              'compiler, simulator or neural execution.'), indent=2)+'\n')
    sdk = json.loads((candidate / 'manifest.json').read_text())['sdk_sha256']
    assert sdk == json.loads((baseline / 'manifest.json').read_text())['sdk_sha256']
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=sdk,
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path.insert(0, str(root / 'tools'))
    from executor import verify, sha
    from model_partition_contract import audit
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    for name, expected in config['execution_sha256'].items():
        assert sha(root / name / 'execution.json') == expected
    print('inspect-whole-ELF-contract', flush=True)
    report = audit(root / 'candidate', root / 'baseline')
    report.update(success=True, offline_only=True)
    (root / 'results.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--candidate', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'support/model_partition')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.candidate:
        if args.baseline is None:
            parser.error('--baseline required')
        prepare(args.candidate.resolve(), args.baseline.resolve(), args.source_root.resolve(),
                args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 180, rss_limit_kib=2*1024*1024)

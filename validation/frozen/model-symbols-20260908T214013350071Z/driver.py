"""Bounded actual ELF symbol qualification of a completed full-layout audit."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(audited, source, project):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import verify, sha
    manifest = verify(audited)
    execution = json.loads((audited / 'execution.json').read_text())
    result = json.loads((audited / 'results.json').read_text())
    assert execution['success'] and result['success'] and result['offline_only']
    assert execution['manifest_sha256'] == sha(audited / 'manifest.json')
    assert execution['results_sha256'] == sha(audited / 'results.json')
    for name, expected in result['report_sha256'].items():
        assert sha(audited / name) == expected
        assert json.loads((audited / name).read_text())['passed']
    out = project / 'evidence' / ('model-symbols-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    shutil.copytree(audited, out / 'audited', ignore=shutil.ignore_patterns('__pycache__'))
    for name in ['tools/model_symbols.py', 'tools/sdk_probe.py',
                 'src/csl_llm/__init__.py', 'src/csl_llm/decoder_layout.py', 'src/csl_llm/regions.py']:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    (out / 'config.json').write_text(json.dumps(dict(
        source_execution_sha256=sha(audited / 'execution.json'),
        timeout_seconds=180, rss_limit_kib=2*1024*1024,
        scope='Full-layout actual ELF symbols only. No compiler, core, SDK runtime '
              'or numerical execution; selected original weights remain partial.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path.insert(0, str(root / 'tools'))
    from executor import verify, sha
    from model_symbols import audit
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    assert sha(root / 'audited/execution.json') == config['source_execution_sha256']
    verify(root / 'audited')
    print('audit-actual-full-model-symbols', flush=True)
    symbols = audit(root / 'audited/compiled/out/bin')
    (root / 'symbols.json').write_text(json.dumps(symbols, indent=2)+'\n')
    (root / 'results.json').write_text(json.dumps(dict(success=True, offline_only=True,
        source_execution_sha256=config['source_execution_sha256'],
        symbols_sha256=sha(root / 'symbols.json'), application_coordinates=40530,
        scope=config['scope']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--audit', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.audit:
        prepare(args.audit.resolve(), args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 180, rss_limit_kib=2*1024*1024)

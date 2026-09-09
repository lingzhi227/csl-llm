"""Freeze completed full-model compilation for bounded offline ELF audits."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(compiled, source, project):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import verify, sha
    verify(source)
    manifest = verify(compiled)
    execution = json.loads((compiled / 'execution.json').read_text())
    result = json.loads((compiled / 'results.json').read_text())
    assert execution['success'] and result['success'] and result['compile_only']
    assert execution['manifest_sha256'] == sha(compiled / 'manifest.json')
    assert execution['results_sha256'] == sha(compiled / 'results.json')
    assert all(value is None for value in execution['after_cleanup_identities'].values())
    out = project / 'evidence' / ('model-elf-audit-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    target = out / 'compiled'
    target.mkdir()
    # Preserve the original input manifest and completion receipts unchanged.
    # The outer audit manifest additionally binds all generated compiler files.
    names = set(manifest['files']) | {'manifest.json', 'execution.json', 'results.json'}
    for name in names:
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(compiled / name, destination)
    shutil.copytree(compiled / 'out', target / 'out')
    for directory in ('tools', 'src'):
        shutil.copytree(source / directory, out / directory, ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source / 'manifest.json', out / 'source-auditor-manifest.json')
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    (out / 'config.json').write_text(json.dumps(dict(
        source_compile=compiled.name, source_execution_sha256=sha(compiled / 'execution.json'),
        source_auditor_manifest_sha256=sha(source / 'manifest.json'),
        timeout_seconds=600, rss_limit_kib=4*1024*1024,
        scope='Offline initial ELF bytes, static SRAM and DSR allocations. No compiler, simulator, runtime identity, full original model or neural execution.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path.insert(0, str(root / 'tools'))
    from executor import verify, sha
    from model_initializer_audit import audit
    from elf_memory import inspect
    import dsr_graph
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    assert sha(root / 'source-auditor-manifest.json') == config['source_auditor_manifest_sha256']
    sources = json.loads((root / 'source-auditor-manifest.json').read_text())
    assert all(sha(root / name) == expected for name, expected in sources['files'].items())
    def save(name, value):
        with (root / name).open('x') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
    compiled = root / 'compiled'
    assert sha(compiled / 'execution.json') == config['source_execution_sha256']
    print('audit-initializers', flush=True)
    initializers = audit(compiled)
    save('initializers.json', initializers)
    print('audit-static-memory', flush=True)
    rows = [inspect(path) for path in sorted((compiled / 'out/bin').glob('*.elf'))]
    assert rows
    memory = dict(passed=True, classes=rows,
        max_static_high_water_bytes=max(row['static_high_water_bytes'] for row in rows),
        manifest_sha256=sha(compiled / 'manifest.json'),
        scope='Every allocated ELF section classified; static SRAM only, no dynamic lifetime claim.')
    save('elf-memory.json', memory)
    print('audit-dsr', flush=True)
    dsr = dsr_graph.report(compiled)
    save('dsr-graphs.json', dsr)
    (root / 'results.json').write_text(json.dumps(dict(success=True, offline_only=True,
        source_execution_sha256=config['source_execution_sha256'],
        report_sha256={name: sha(root / name) for name in
                       ['initializers.json', 'elf-memory.json', 'dsr-graphs.json']},
        scope=config['scope']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-compile', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--audit-source', type=Path)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.from_compile:
        if args.audit_source is None:
            parser.error('--audit-source required')
        prepare(args.from_compile.resolve(), args.audit_source.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 600, rss_limit_kib=4*1024*1024)

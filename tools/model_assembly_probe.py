"""Freeze complete original-model audits for guarded intact-ELF assembly.

No compiler or simulator is started. Incomplete collections are rejected before
snapshotting. Static assembly does not qualify full-model neural execution.
"""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def prepare(audits, source, project):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import verify, sha
    from model_partition_assemble import complete_coverage
    if len(set(audits)) != len(audits):
        raise ValueError('Repeated source audit')
    configs = [json.loads((root / 'compiled/static-input-config.json').read_text())
               for root in audits]
    complete_coverage(configs)
    sdk_hash = None
    records = []
    for index, root in enumerate(audits):
        manifest = verify(root)
        execution = json.loads((root / 'execution.json').read_text())
        result = json.loads((root / 'results.json').read_text())
        assert execution['success'] and result['success'] and result['offline_only']
        assert execution['manifest_sha256'] == sha(root / 'manifest.json')
        assert execution['results_sha256'] == sha(root / 'results.json')
        if sdk_hash is None:
            sdk_hash = manifest['sdk_sha256']
        assert manifest['sdk_sha256'] == sdk_hash == execution['sdk_sha256']
        records.append(dict(path=f'audits/{index}', source_bundle=root.name,
                            execution_sha256=sha(root / 'execution.json')))
    out = project / 'evidence' / ('model-assembled-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    for root, row in zip(audits, records):
        shutil.copytree(root, out / row['path'], ignore=shutil.ignore_patterns('__pycache__'))
    names = ['tools/model_partition_assemble.py', 'tools/model_partition_contract.py',
             'tools/sdk_probe.py', 'tools/elf_partition.py', 'tools/compare_elf_images.py',
             'tools/model_symbols.py', 'tools/decoder_chain_symbols.py', 'tools/sdk_core.py',
             'src/csl_llm/__init__.py', 'src/csl_llm/decoder_layout.py', 'src/csl_llm/regions.py']
    for name in names:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    (out / 'config.json').write_text(json.dumps(dict(audits=records,
        timeout_seconds=1800, rss_limit_kib=8*1024*1024,
        scope='Full original-model static intact-ELF assembly and actual diagnostic '
              'symbol qualification. No compiler, SdkRuntime or neural execution.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=sdk_hash,
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path.insert(0, str(root / 'tools'))
    from executor import verify, sha
    from model_partition_assemble import assemble
    from model_symbols import audit as audit_symbols
    from sdk_core import contract_symbols
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    audits = []
    for row in config['audits']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root / relative
        assert sha(path / 'execution.json') == row['execution_sha256']
        audits.append(path)
    print('validate-and-compose-complete-original-model', flush=True)
    result = assemble(audits, root / 'out')
    print('audit-complete-assembly-diagnostic-symbols', flush=True)
    symbols = audit_symbols(root / 'out/bin', diagnostics=True)
    contract_symbols(symbols, result['composition'])
    (root / 'symbols.json').write_text(json.dumps(symbols, indent=2)+'\n')
    result['symbols_sha256'] = sha(root / 'symbols.json')
    result['scope'] = config['scope']
    result.update(success=True)
    (root / 'results.json').write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--audit', action='append', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'support/model_assembly')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.prepare:
        if not args.audit:
            parser.error('--audit required')
        prepare([p.resolve() for p in args.audit], args.source_root.resolve(),
                args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 1800, rss_limit_kib=8*1024*1024)

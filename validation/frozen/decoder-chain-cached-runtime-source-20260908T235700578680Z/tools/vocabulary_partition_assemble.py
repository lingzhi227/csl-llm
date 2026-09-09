"""Compose all9496 original vocabulary tiles from bounded compiler batches.

No compiler, simulator, ELF modification or neural computation runs here.
Each source batch remains independently frozen and must have completed audits.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_batch(path):
    from executor import verify
    manifest = verify(path)
    config = json.loads((path / 'config.json').read_text())
    execution = json.loads((path / 'execution.json').read_text())
    result = json.loads((path / 'results.json').read_text())
    assert execution['success'] and result['success'] and result['compile_only']
    assert execution['manifest_sha256'] == digest(path / 'manifest.json')
    assert execution['results_sha256'] == digest(path / 'results.json')
    assert config['representation'] == 'bytes' and config['matrix_tiles'] == 9496
    assert result['all_matrix_coordinates_exact']
    indices = config['static_tile_indices']
    assert 0 < len(indices) <= 1024 and len(set(indices)) == len(indices)
    assert all(type(index) is int and 0 <= index < 9496 for index in indices)
    assert result['static_original_tiles'] == len(indices)
    assert result['zero_matrix_tiles'] == 9496-len(indices)
    resources = json.loads((path / 'elf-memory.json').read_text())
    dsr = json.loads((path / 'dsr-graphs.json').read_text())
    for report in [resources, dsr]:
        assert report['passed'] and report['manifest_sha256'] == digest(path / 'manifest.json')
    files = {p.name: digest(p) for p in (path / 'out/bin').glob('*.elf')}
    assert files == {row['elf']: row['sha256'] for row in resources['classes']}
    assert files == dsr['elf_sha256']
    assert resources['max_static_high_water_bytes'] <= 49152
    for row in result['images']:
        assert files[row['file']] == row['sha256']
    layout = (path / 'layout.csl').read_text()
    selected = set()
    for line in layout.splitlines():
        if line.startswith('@set_tile_code') and '.static_weights=true' in line:
            assert '.matrix=true' in line
            x, y = map(int, re.match(r'@set_tile_code\((\d+),(\d+),', line).groups())
            assert x < 192 and f'.weight_file="weights_{x}_{y}.csl"' in line
            selected.add(y*192+x)
    assert selected == set(indices)
    normalized = re.sub(r'\.weight_file="weights_\d+_\d+\.csl"', '.weight_file="<empty>"', layout)
    normalized = normalized.replace('.static_weights=true', '.static_weights=false')
    common_sources = {name: sha for name, sha in manifest['files'].items()
                      if name.endswith('.csl') and name != 'layout.csl' and not name.startswith('weights_')}
    return config, normalized, common_sources


def prepare(batches, source, project):
    out = project / 'evidence' / ('vocabulary-assembled-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    # The pinned SDK wrapper mounts only its working directory. Keep all
    # dependencies inside this bundle, with independent copies and frozen
    # hashes; never rely on an undeclared external bind or writable hardlink.
    records = []
    snapshot_start = time.monotonic()
    for index, batch in enumerate(batches):
        destination = out / 'batches' / f'batch-{index}'
        shutil.copytree(batch, destination)
        records.append(dict(path=str(destination.relative_to(out)),
            source_path=str(batch), source_bundle=batch.name,
            files={name: digest(destination / name) for name in [
                'manifest.json', 'execution.json', 'results.json', 'elf-memory.json', 'dsr-graphs.json']}))
    (out / 'preparation.json').write_text(json.dumps(dict(
        snapshot_seconds=time.monotonic()-snapshot_start,
        snapshot_bytes=sum(p.stat().st_size for p in (out / 'batches').rglob('*') if p.is_file()),
        scope='Batch snapshot copying and receipt hashing only, before final manifest hashing. Separate from guarded assembly-worker wall time; not compiler or inference performance.'), indent=2)+'\n')
    for path, name in [(Path(__file__), 'driver.py'), (source / 'tools/sdk_probe.py', 'executor.py'),
                       (source / 'tools/elf_partition.py', 'elf_partition.py'),
                       (source / 'tools/vocabulary_symbols.py', 'vocabulary_symbols.py'),
                       (source / 'tools/compare_elf_images.py', 'compare_elf_images.py')]:
        shutil.copy2(path, out / name)
    (out / 'batches.json').write_text(json.dumps(records, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
        files={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    from executor import verify
    from elf_partition import compose, image
    verify(root)
    os.chdir(root)
    records = json.loads((root / 'batches.json').read_text())
    parts = []
    seen = set()
    reference = None
    bindings = []
    application = {(x+4, y+1) for y in range(50) for x in range(193)}
    matrix = {(index % 192+4, index//192+1) for index in range(9496)}
    guards = application-matrix

    def guard_images(directory):
        values = {}
        for path in (directory / 'out/bin').glob('*.elf'):
            value = image(path)
            for point in value['coordinates']:
                point = tuple(point)
                if point in guards:
                    assert point not in values
                    values[point] = value
        assert set(values) == guards
        return values

    for row in records:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        batch = root / relative
        for name, sha in row['files'].items():
            assert digest(batch / name) == sha, (batch.name, name)
        config, normalized, common = validate_batch(batch)
        invariant = (config['parent_manifest_sha256'], config['source_packed_sha256'], normalized, common)
        if reference is None:
            reference = invariant
            first = batch
            common_guards = guard_images(batch)
        else:
            assert invariant == reference, 'Batch program/layout or original source differs'
            assert guard_images(batch) == common_guards, 'Common controller/guard loader images differ'
        indices = set(config['static_tile_indices'])
        assert not indices & seen, 'Repeated original matrix tile'
        seen |= indices
        parts.append((batch / 'out', {(i % 192+4, i//192+1) for i in indices}))
        bindings.append(dict(bundle=row['source_bundle'], snapshot=str(relative),
                             indices=sorted(indices), evidence_hashes=row['files']))
    assert seen == set(range(9496)), 'Every original vocabulary tile is required'
    parts.append((first / 'out', guards))
    report = compose(parts, root / 'out', application)
    (root / 'composition.json').write_text(json.dumps(report, indent=2)+'\n')
    from vocabulary_symbols import audit
    symbols = audit(root / 'out/bin')
    (root / 'symbols.json').write_text(json.dumps(symbols, indent=2)+'\n')
    (root / 'results.json').write_text(json.dumps(dict(success=True, compile_only=True,
        original_tiles=9496, application_coordinates=9650, source_packed_sha256=reference[1],
        composition_sha256=digest(root / 'composition.json'), batches=bindings,
        symbols_sha256=digest(root / 'symbols.json'),
        scope='Complete original vocabulary static compiler-artifact composition from audited batches. '
              'Intact ELF copies, original tile coverage, common program/guards/RPC/I/O and loader '
              'coordinate checks only. No SdkRuntime, full vocabulary numerical or model inference acceptance.'), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--batch', type=Path, action='append')
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.prepare:
        if not args.batch:
            parser.error('--batch is required')
        prepare([p.resolve() for p in args.batch], args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 300, rss_limit_kib=4*1024*1024)

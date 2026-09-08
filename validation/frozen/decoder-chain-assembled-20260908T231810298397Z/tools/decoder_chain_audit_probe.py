"""Actual ELF initializer, SRAM and DSR checks for the native two-layer gate."""
import argparse
import datetime
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]


def expected_symbols(compiled):
    from model_initializer_audit import expected_symbols as full_model_expected
    config = json.loads((compiled / 'static-input-config.json').read_text())
    if not config['matrices'] or any(row['original_record']['layer'] not in (0, 1)
                                     for row in config['matrices']):
        raise ValueError('Only bounded original layer0/1 matrices are valid')
    # Original coordinates and declarations are unchanged in the first two
    # regions. The original generator validates each selected declaration/hash.
    for (x, y), fields in full_model_expected(compiled):
        if y >= 20:
            break
        if x < 128:
            yield (x, y), fields
    for y in range(1, 20):
        yield (128, y), {name: bytes(4) for name in ('weights', 'norms', 'frequency', 'bias')}
    yield (128, 0), dict(input=bytes(3584), output=bytes(3584),
                        progress=bytes(4), status=bytes(8), timing=bytes(12))


def audit(compiled):
    from sdk_probe import verify, sha
    from model_symbols import resolve
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    verify(compiled)
    config = json.loads((compiled / 'config.json').read_text())
    execution = json.loads((compiled / 'execution.json').read_text())
    result = json.loads((compiled / 'results.json').read_text())
    assert execution['success'] and result['success'] and result['compile_only']
    assert execution['manifest_sha256'] == sha(compiled / 'manifest.json')
    assert execution['results_sha256'] == sha(compiled / 'results.json')
    assert (config['width'], config['height'], config['weight_storage']) == (129, 20, 'u8')
    assert config['layers'] == [0, 1] and config['matrix_tiles'] == 2088
    expected = dict(expected_symbols(compiled))
    assert set(expected) == {(x, y) for y in range(20) for x in range(129)}
    seen = set()
    images = []
    for path in sorted((compiled / 'out/bin').glob('*.elf')):
        memory = ELFMemory(str(path))
        assert tuple(memory.get_fabric_dimensions()) == (136, 22)
        with path.open('rb') as stream:
            elf = ELFFile(stream)
            assert elf.little_endian and elf.elfclass == 64
            table = elf.get_section_by_name('.symtab')
            assert table is not None
            objects = [dict(name=s.name, address=int(s['st_value']), size=int(s['st_size']))
                       for s in table.iter_symbols() if s['st_info']['type'] == 'STT_OBJECT' and s['st_size']]
        coordinates = []
        for px, py in memory.iter_coordinates():
            point = (px-4, py-1)
            if point not in expected or point in seen:
                raise ValueError('Unexpected or repeated application coordinate')
            seen.add(point)
            coordinates.append(list(point))
            for name, raw in expected[point].items():
                symbol = resolve(objects, name, len(raw))
                actual = memory.read_data(px, py, symbol['address'], symbol['size'])
                if any(value is None for value in actual) or bytes(actual) != raw:
                    raise ValueError(f'Actual loaded initializer differs: {point}, {name}')
        if not coordinates:
            raise ValueError('Empty application ELF')
        images.append(dict(elf=path.name, sha256=sha(path), coordinates=coordinates))
    assert seen == set(expected)
    assert sum(len(fields.get('weights', b'')) == 28672 for fields in expected.values()) == 2088
    return dict(passed=True, application_coordinates=2580, matrix_tiles=2088,
        selected_matrix_tiles=config['static_input_binding']['selected_matrix_tiles'],
        initialized_controllers=2,
        initialized_bias_roots=config['static_input_binding']['initialized_bias_roots'],
        images=images, manifest_sha256=sha(compiled / 'manifest.json'),
        execution_sha256=sha(compiled / 'execution.json'),
        scope='Actual loaded ELF initializers and complete two-layer placement only; '
              'original checkpoint review remains separate, no SdkRuntime or neural execution.')


def prepare(compiled, source, project):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import verify, sha
    manifest = verify(compiled)
    execution = json.loads((compiled / 'execution.json').read_text())
    result = json.loads((compiled / 'results.json').read_text())
    assert execution['success'] and result['success'] and result['compile_only']
    assert execution['manifest_sha256'] == sha(compiled / 'manifest.json')
    assert execution['results_sha256'] == sha(compiled / 'results.json')
    out = project / 'evidence' / ('decoder-chain-audit-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    shutil.copytree(compiled, out / 'compiled', ignore=shutil.ignore_patterns('__pycache__'))
    names = ['tools/model_initializer_audit.py', 'tools/model_symbols.py', 'tools/sdk_probe.py',
             'tools/elf_memory.py', 'tools/dsr_graph.py', 'src/csl_llm/__init__.py',
             'src/csl_llm/decoder_layout.py', 'src/csl_llm/regions.py']
    for name in names:
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    (out / 'config.json').write_text(json.dumps(dict(source_execution_sha256=sha(compiled / 'execution.json'),
        timeout_seconds=600, rss_limit_kib=4*1024*1024,
        scope='Two-layer actual ELF initializers and static resources only; no SDK execution.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path.insert(0, str(root / 'tools'))
    sys.path.insert(0, str(root / 'src'))
    from executor import verify, sha
    from elf_memory import inspect
    from dsr_graph import report
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    compiled = root / 'compiled'
    assert sha(compiled / 'execution.json') == config['source_execution_sha256']
    print('audit-native-two-layer-initializers', flush=True)
    initializers = audit(compiled)
    (root / 'initializers.json').write_text(json.dumps(initializers, indent=2)+'\n')
    print('audit-native-two-layer-resources', flush=True)
    rows = [inspect(p) for p in sorted((compiled / 'out/bin').glob('*.elf'))]
    memory = dict(passed=True, classes=rows,
        max_static_high_water_bytes=max(row['static_high_water_bytes'] for row in rows),
        manifest_sha256=sha(compiled / 'manifest.json'))
    (root / 'elf-memory.json').write_text(json.dumps(memory, indent=2)+'\n')
    (root / 'dsr-graphs.json').write_text(json.dumps(report(compiled), indent=2)+'\n')
    (root / 'results.json').write_text(json.dumps(dict(success=True, offline_only=True,
        source_execution_sha256=config['source_execution_sha256'],
        report_sha256={name: sha(root / name) for name in ('initializers.json', 'elf-memory.json', 'dsr-graphs.json')},
        scope=config['scope']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-compile', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.from_compile:
        prepare(args.from_compile.resolve(), args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 600, rss_limit_kib=4*1024*1024)

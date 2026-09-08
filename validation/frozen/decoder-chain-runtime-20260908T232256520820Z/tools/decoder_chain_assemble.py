"""Complete original two-layer composition and actual loaded-byte verification.

Retain the bounded partition auditor unchanged. Merge only each partition's
declared original owners, check common initializers for equality, and verify
every assembled ELF against the resulting complete 2,088-matrix expectation.
"""
import json
from pathlib import Path


def ownership():
    from csl_llm.decoder_layout import tiles
    matrix = {(tile.x, tile.y) for layer in (0, 1) for tile in tiles(layer)
              if tile.operation != 'kv'}
    universe = {(x, y) for y in range(20) for x in range(129)}
    assert len(matrix) == 2088 and len(universe-matrix) == 492
    return matrix, universe


def complete_coverage(configs):
    matrix, universe = ownership()
    indices, points = set(), set()
    provenance = None
    for config in configs:
        mapping = config['mapping_audit']
        if mapping['passed'] is not True or mapping['matrix_tiles'] != 34552:
            raise ValueError('Original full-model pack mapping must be verified')
        current = (config['original_pack'], mapping['pack_manifest_sha256'],
                   mapping['pack_verification_sha256'], config['frequency_fixture_manifest_sha256'],
                   config['frequency_auxiliary_sha256'])
        if provenance is None:
            provenance = current
        elif provenance != current:
            raise ValueError('Partition original provenance differs')
        if not 1 <= len(config['matrices']) <= 1024:
            raise ValueError('Each original partition must remain bounded')
        for row in config['matrices']:
            original = row['original_record']
            offset = original['offset']
            point = tuple(row['coordinate'])
            if (type(offset) is not int or offset < 0 or offset % 28672 or
                    offset // 28672 >= 2088 or original['layer'] not in (0, 1) or
                    point != (original['x'], original['y']) or point not in matrix):
                raise ValueError('Invalid original two-layer matrix owner')
            if offset // 28672 in indices or point in points:
                raise ValueError('Repeated original matrix or physical owner')
            indices.add(offset // 28672)
            points.add(point)
    if indices != set(range(2088)) or points != matrix:
        raise ValueError('All 2088 original matrices are required')
    return matrix, universe


def complete_expected(compiled_roots):
    from decoder_chain_audit_probe import expected_symbols
    configs = [json.loads((root / 'static-input-config.json').read_text()) for root in compiled_roots]
    matrix, universe = complete_coverage(configs)
    common, merged = universe-matrix, {}
    common_expected = None
    for root, config in zip(compiled_roots, configs):
        expected = dict(expected_symbols(root))
        if set(expected) != universe:
            raise ValueError('Incomplete partition initializer expectation')
        current = {point: expected[point] for point in common}
        if common_expected is None:
            common_expected = current
        elif common_expected != current:
            raise ValueError('Common original initializers differ')
        for row in config['matrices']:
            point = tuple(row['coordinate'])
            merged[point] = expected[point]
    merged.update(common_expected)
    assert set(merged) == universe
    assert sum(len(fields.get('weights', b'')) == 28672 for fields in merged.values()) == 2088
    return merged


def audit_loaded(directory, expected):
    from sdk_probe import sha
    from model_symbols import resolve
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    seen, images = set(), []
    for path in sorted(directory.glob('*.elf')):
        memory = ELFMemory(str(path))
        assert tuple(memory.get_fabric_dimensions()) == (136, 22)
        with path.open('rb') as stream:
            elf = ELFFile(stream)
            assert elf.little_endian and elf.elfclass == 64
            table = elf.get_section_by_name('.symtab')
            assert table is not None
            objects = [dict(name=s.name, address=int(s['st_value']), size=int(s['st_size']))
                       for s in table.iter_symbols() if s['st_info']['type'] == 'STT_OBJECT' and s['st_size']]
        points = []
        for px, py in memory.iter_coordinates():
            point = (px-4, py-1)
            if point not in expected or point in seen:
                raise ValueError('Invalid assembled coordinate ownership')
            seen.add(point)
            points.append(list(point))
            for logical, raw in expected[point].items():
                symbol = resolve(objects, logical, len(raw))
                actual = memory.read_data(px, py, symbol['address'], len(raw))
                if any(value is None for value in actual) or bytes(actual) != raw:
                    raise ValueError(f'Assembled original initializer differs: {point} {logical}')
        if not points:
            raise ValueError('Empty assembled application ELF')
        images.append(dict(elf=path.name, sha256=sha(path), coordinates=points))
    if seen != set(expected):
        raise ValueError('Incomplete assembled original initializer coverage')
    return dict(passed=True, application_coordinates=len(seen), original_matrix_tiles=2088,
                images=images, scope='Actual assembled ELF loaded initial bytes; no SDK execution.')


def assemble(audits, output):
    from sdk_probe import sha
    from model_partition_contract import audit_partition_geometry
    from elf_partition import compose
    roots = [Path(root).resolve() for root in audits]
    if not roots or len(set(roots)) != len(roots) or output.exists():
        raise ValueError('Unique audits and fresh output required')
    configs = [json.loads((root / 'compiled/static-input-config.json').read_text()) for root in roots]
    matrix, universe = complete_coverage(configs)
    parts, contracts = [], []
    for root, config in zip(roots, configs):
        contract = audit_partition_geometry(root, roots[0], matrix, universe)
        selected = {tuple(row['coordinate']) for row in config['matrices']}
        actual = {tuple(point) for image in contract['classes']['selected'] for point in image['coordinates']}
        if actual != selected:
            raise ValueError('Actual original partition ownership differs')
        parts.append((root / 'compiled/out', {(x+4, y+1) for x, y in selected}))
        contracts.append(dict(source_audit=root.name, execution_sha256=sha(root / 'execution.json'),
                              initializer_sha256=sha(root / 'initializers.json'), contract=contract))
    expected = complete_expected([root / 'compiled' for root in roots])
    parts.append((roots[0] / 'compiled/out', {(x+4, y+1) for x, y in universe-matrix}))
    composition = compose(parts, output, {(x+4, y+1) for x, y in universe})
    initializers = audit_loaded(output / 'bin', expected)
    return dict(passed=True, offline_only=True, contracts=contracts,
                composition=composition, initializers=initializers,
                scope='Complete original two-layer intact ELF composition and actual initial bytes. '
                      'Runtime, neural correctness and full-model acceptance remain separate.')

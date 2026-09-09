"""Read actual SDK ELF data and coordinates; check every initialized MLP tile.

Run with the pinned SDK cs_python. This tool never writes program memory or
modifies an ELF. The report is separate from runtime numerical acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(bundle):
    import numpy as np
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, expected in manifest['files'].items():
        assert digest(bundle / name) == expected, name
    config = json.loads((bundle / 'config.json').read_text())
    assert config['static_weights'] is True and config['initializer_tiles'] == 916
    inputs = np.load(bundle / 'inputs.npz')
    expected_tiles = {}
    for name, offset in [('up', 0), ('gate', 16), ('down', 32)]:
        values = inputs['weights_' + name]
        assert values.dtype.kind == 'u' and values.dtype.itemsize == 2
        for y in range(values.shape[0]):
            for x in range(values.shape[1]):
                expected_tiles[x+offset, y] = values[y, x].astype('<u2').tobytes(order='C')
    for (x, y), raw in expected_tiles.items():
        source = (bundle / f'weights_{x}_{y}.csl').read_text()
        match = re.fullmatch(r'const values=\[7168\]u32\{(.*)\};\n', source)
        assert match is not None, ('initializer syntax', x, y)
        words = match[1].split(',')
        assert len(words) == 7168 and all(re.fullmatch(r'0x[0-9a-f]{8}', word) for word in words)
        assert struct.pack('<7168I', *(int(word, 16) for word in words)) == raw, ('CSL initializer differs', x, y)
    seen, reports = set(), []
    for file in sorted((bundle / 'out/bin').glob('*.elf')):
        coordinates = [(x-4, y-1) for x, y in ELFMemory(str(file)).iter_coordinates()]
        selected = [point for point in coordinates if point in expected_tiles]
        if not selected:
            continue
        with file.open('rb') as stream:
            elf = ELFFile(stream)
            assert elf.little_endian
            symbols = elf.get_section_by_name('.symtab').get_symbol_by_name('weights')
            assert symbols is not None and len(symbols) == 1
            symbol = symbols[0]
            assert symbol['st_size'] == 28672 and isinstance(symbol['st_shndx'], int)
            section = elf.get_section(symbol['st_shndx'])
            assert section['sh_type'] == 'SHT_PROGBITS'
            offset = int(symbol['st_value']) - int(section['sh_addr'])
            assert 0 <= offset and offset + 28672 <= section['sh_size']
            raw = section.data()[offset:offset+28672]
        for point in selected:
            assert point not in seen, ('duplicate', point)
            assert raw == expected_tiles[point], ('initialized bytes differ', file.name, point)
            seen.add(point)
        reports.append(dict(file=file.name, sha256=digest(file), coordinates=selected,
                            weights_address=int(symbol['st_value']), weights_bytes=len(raw),
                            weights_sha256=hashlib.sha256(raw).hexdigest()))
    assert seen == set(expected_tiles) and len(seen) == 916
    return dict(passed=True, initialized_tiles=len(seen), payload_bytes=len(seen)*28672,
                elfs=reports, manifest_sha256=digest(bundle / 'manifest.json'),
                tool_sha256=digest(Path(__file__)),
                scope='All916 generated CSL literal arrays, tile coordinates and actual compiled ELF initialized bytes equal frozen raw BF16 fixture tiles, including padding. No ELF writes. Not a runtime completion, dynamic stack or full-model initialization proof.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    result = audit(args.bundle.resolve())
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('INITIALIZED TILES', result['initialized_tiles'])

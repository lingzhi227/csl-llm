"""Read-only original tied matrix → generated literals → SDK ELF ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(bundle, model):
    import numpy as np
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    config = json.loads((bundle / 'config.json').read_text())
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        assert digest(bundle / name) == sha, name
    assert digest(model) == config['model_sha256']
    with model.open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    tensor = header['model.embed_tokens.weight']
    assert tensor['dtype'] == 'BF16' and tensor['shape'] == [151936, 896]
    assert tensor['data_offsets'][1]-tensor['data_offsets'][0] == 151936*896*2
    weights = np.memmap(model, mode='r', dtype='<u2', offset=8+header_length+tensor['data_offsets'][0], shape=(151936, 896))
    expected = {(x, y) for y in range(50) for x in range(192) if y*24+x//8 < 1187}
    assert len(expected) == 9496
    seen, records = set(), []
    for file in sorted((bundle / 'out/bin').glob('*.elf')):
        coordinates = [(x-4, y-1) for x, y in ELFMemory(str(file)).iter_coordinates()]
        selected = [point for point in coordinates if point in expected]
        if not selected:
            continue
        with file.open('rb') as stream:
            elf = ELFFile(stream)
            assert elf.little_endian
            entries = elf.get_section_by_name('.symtab').get_symbol_by_name('weights')
            assert entries and len(entries) == 1
            symbol = entries[0]
            assert symbol['st_size'] == 28672 and isinstance(symbol['st_shndx'], int)
            section = elf.get_section(symbol['st_shndx'])
            assert section['sh_type'] == 'SHT_PROGBITS'
            offset = int(symbol['st_value'])-int(section['sh_addr'])
            assert 0 <= offset and offset+28672 <= section['sh_size']
            actual = section.data()[offset:offset+28672]
            for point in selected:
                assert point not in seen
                seen.add(point)
                x, y = point
                row, column = (y*24+x//8)*128, (x%8)*112
                payload = weights[row:row+128, column:column+112].T.copy().tobytes()
                source = bundle / f'weights_{x}_{y}.csl'
                text = source.read_text()
                match = re.fullmatch(r'const values=\[7168\]u32\{([^}]+)\};\n', text)
                assert match is not None
                words = match[1].split(',')
                assert len(words) == 7168 and all(re.fullmatch(r'0x[0-9a-f]{8}', word) for word in words)
                literal = struct.pack('<7168I', *(int(word, 16) for word in words))
                assert payload == literal == actual, (point, file.name)
                records.append(dict(coordinate=point, vocabulary_row=row, inner_column=column,
                                    elf=file.name, elf_sha256=digest(file), source_sha256=digest(source),
                                    weight_address=int(symbol['st_value']), payload_sha256=hashlib.sha256(payload).hexdigest()))
    assert seen == expected
    return dict(passed=True, matrix_tiles=len(seen), original_weight_bytes=151936*896*2,
                records=records, model_sha256=config['model_sha256'], manifest_sha256=digest(bundle / 'manifest.json'),
                tool_sha256=digest(Path(__file__)),
                scope='Every original tied embedding BF16 word, all9496 normal CSL literals and actual ELF initialized weight bytes at SDK coordinates. No runtime/stack/fullmodel acceptance claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    value = audit(args.bundle.resolve(), args.model.resolve())
    with args.report.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print('INITIALIZED VOCABULARY TILES', value['matrix_tiles'])

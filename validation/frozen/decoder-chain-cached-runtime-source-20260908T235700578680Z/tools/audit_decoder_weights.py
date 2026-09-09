"""Read-only checkpoint → CSL literals → actual ELF initialized-byte audit."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def literal(path, name, count):
    match = re.search(r'const '+name+rf'=\[{count}\]u32\{{([^}}]+)\}};', path.read_text())
    assert match is not None, (path.name, name)
    words = match[1].split(',')
    assert len(words) == count and all(re.fullmatch(r'0x[0-9a-f]{8}', word) for word in words)
    return struct.pack('<'+str(count)+'I', *(int(word, 16) for word in words))


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
    def tensor(name):
        item = header[name]
        assert item['dtype'] == 'BF16'
        return np.memmap(model, mode='r', dtype='<u2', offset=8+header_length+item['data_offsets'][0], shape=tuple(item['shape']))
    inputs = np.load(bundle / 'inputs.npz')
    expected = {}
    layer = config.get('layer', 0)
    assert isinstance(layer, int) and not isinstance(layer, bool) and 0 <= layer < 24
    prefix = f'model.layers.{layer}.'
    for name in inputs.files:
        match = re.fullmatch(r'weights_(\d+)_(\d+)', name)
        if match is None:
            continue
        x, y = map(int, match.groups())
        if x < 32 and y < 19:
            op, category = ('up' if x < 16 else 'gate'), 'mlp'
            row, col = (2*y+(x % 16)//8)*128, (x % 8)*112
        elif 32 <= x < 54 and y < 14:
            op, category, row, col = 'down', 'mlp', (y//2)*128, ((y % 2)*22+x-32)*112
        else:
            assert 54 <= x < 62 and y < 16
            op, category = ('q' if y < 7 else 'k' if y == 7 else 'v' if y == 8 else 'o'), 'self_attn'
            row, col = (y if y < 7 else 0 if y < 9 else y-9)*128, (x-54)*112
        raw = tensor(prefix+f'{category}.{op}_proj.weight')
        valid = min(112, raw.shape[1]-col)
        tile = np.zeros((112, 128), dtype='<u2')
        tile[:valid] = raw[row:row+128, col:col+valid].T
        np.testing.assert_array_equal(tile, inputs[name])
        payload = tile.tobytes(order='C')
        assert literal(bundle / (name+'.csl'), 'values', 7168) == payload
        expected[x, y] = {'weights': payload}
    assert len(expected) == 1044
    for y in range(9):
        name = 'q' if y < 7 else 'k' if y == 7 else 'v'
        group = y if y < 7 else 0
        payload = tensor(prefix+f'self_attn.{name}_proj.bias')[group*128:(group+1)*128].tobytes()
        assert literal(bundle / f'bias_{name}_{group}.csl', 'bias', 64) == payload
        expected[54, y]['bias'] = payload
    norms = np.concatenate([tensor(prefix+name+'.weight') for name in ['input_layernorm', 'post_attention_layernorm']]).astype('<u2').tobytes()
    assert literal(bundle / 'controller_aux.csl', 'norms', 896) == norms
    source = (bundle / 'controller_aux.csl').read_text()
    frequency_text = re.search(r'const frequency=\[32\]f32\{([^}]+)\};', source)
    assert frequency_text is not None
    frequencies = re.findall(r'0x[0-9a-f]{8}', frequency_text[1])
    assert len(frequencies) == 32
    frequency = struct.pack('<32I', *(int(word, 16) for word in frequencies))
    expected[62, 0] = {'norms': norms, 'frequency': frequency}
    seen, reports = set(), []
    for file in sorted((bundle / 'out/bin').glob('*.elf')):
        coordinates = [(x-4, y-1) for x, y in ELFMemory(str(file)).iter_coordinates()]
        selected = [point for point in coordinates if point in expected]
        if not selected:
            continue
        with file.open('rb') as stream:
            elf = ELFFile(stream)
            assert elf.little_endian
            symbols = elf.get_section_by_name('.symtab')
            for point in selected:
                assert point not in seen
                seen.add(point)
                fields = {}
                for name, payload in expected[point].items():
                    entries = symbols.get_symbol_by_name(name)
                    assert entries and len(entries) == 1, (file.name, point, name)
                    symbol = entries[0]
                    assert symbol['st_size'] == len(payload) and isinstance(symbol['st_shndx'], int)
                    section = elf.get_section(symbol['st_shndx'])
                    assert section['sh_type'] == 'SHT_PROGBITS'
                    offset = int(symbol['st_value'])-int(section['sh_addr'])
                    assert 0 <= offset and offset+len(payload) <= section['sh_size']
                    raw = section.data()[offset:offset+len(payload)]
                    assert raw == payload, (file.name, point, name)
                    fields[name] = dict(address=int(symbol['st_value']), size=len(raw), sha256=hashlib.sha256(raw).hexdigest())
                reports.append(dict(file=file.name, elf_sha256=digest(file), coordinate=point, fields=fields))
    assert seen == set(expected)
    return dict(passed=True, layer=layer, matrix_tiles=1044, bias_roots=9, norm_controller=1, records=reports,
                model_sha256=config['model_sha256'], manifest_sha256=digest(bundle / 'manifest.json'),
                tool_sha256=digest(Path(__file__)),
                scope='Original checkpoint raw BF16 matrices/biases/norms → all generated literals → actual SDK ELF initialized symbols at all1044 matrix tile coordinates and controller. Frequency literal-to-ELF binding only; frequency derivation is checked by separate numerical/reference evidence. Read-only, not runtime completion or dynamic stack proof.')


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
    print('INITIALIZED DECODER MATRIX TILES', value['matrix_tiles'])

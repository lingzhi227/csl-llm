"""Audit selected full-model static data against actual SDK-loaded ELF bytes.

Compilation is a prerequisite. This verifies initial memory, not initialized
runtime identity, communication, neural execution or full original deployment.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def declaration(text, name, count):
    """Read the exact restricted declarations emitted by model_static_inputs."""
    pattern = rf'const {name}=\[{count}\](u32|f32)\{{([^{{}}]*)\}};'
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise ValueError(f'Missing or repeated declaration: {name}')
    element_type, body = matches[0]
    words = re.findall(r'0x([0-9a-f]{8})', body)
    integers = ','.join('0x'+word for word in words)
    frequencies = ','.join(f'@bitcast(f32,@as(u32,0x{word}))' for word in words)
    if len(words) != count or body != (integers if element_type == 'u32' else frequencies):
        raise ValueError(f'Invalid declaration body: {name}')
    return struct.pack('<'+'I'*count, *(int(word, 16) for word in words))



def expected_symbols(bundle):
    from csl_llm.decoder_layout import tiles, controller_parameters
    config = json.loads((bundle / 'static-input-config.json').read_text())
    selected = {tuple(row['coordinate']): row for row in config['matrices']}
    if not 1 <= len(selected) <= 1024 or len(selected) != len(config['matrices']):
        raise ValueError('Expected a bounded, nonempty, unique matrix selection')
    owners = {(tile.x, tile.y): tile for layer in range(24) for tile in tiles(layer)}
    controllers = {tuple(controller_parameters(layer)['coordinate']): layer for layer in range(24)}
    def is_matrix(point):
        x, y = point
        tile = owners.get(point)
        return (tile is not None and tile.operation != 'kv') or (
            0 <= x < 192 and 160 <= y < 210 and (y-160)*192+x < 9496)

    payloads = {}
    for point, row in selected.items():
        if not is_matrix(point) or row['file'] != f'weights_{point[0]}_{point[1]}.csl':
            raise ValueError('Unknown original matrix owner or declaration filename')
        text = (bundle / row['file']).read_text()
        match = re.fullmatch(r'const values=@get_array\("((?:\\x[0-9a-f]{2})*)"\);\n', text)
        if match is None:
            raise ValueError('Unexpected native byte declaration')
        raw = bytes.fromhex(match[1].replace('\\x', ''))
        if len(raw) != 28672 or hashlib.sha256(raw).hexdigest() != row['original_bytes_sha256']:
            raise ValueError('Original declared matrix digest differs')
        payloads[point] = raw
    zeros = {size: bytes(size) for size in (4, 256, 28672)}
    for y in range(210):
        for x in range(193):
            point = (x, y)
            tile = owners.get(point)
            matrix = is_matrix(point)
            expected = {'weights': payloads.get(point, zeros[28672 if matrix else 4])}
            if point in controllers:
                aux = (bundle / f'controller_aux_{controllers[point]}.csl').read_text()
                expected.update(norms=declaration(aux, 'norms', 896),
                                frequency=declaration(aux, 'frequency', 32))
            elif point == (192, 160):
                expected['norms'] = declaration((bundle / 'final_norm.csl').read_text(), 'norms', 448)
            else:
                expected['norms'] = zeros[4]
                if y < 160:
                    expected['frequency'] = zeros[4]
            if y < 160:
                biased = tile is not None and bool(tile.bias_tensor)
                expected['bias'] = (declaration((bundle / f'bias_{x}_{y}.csl').read_text(), 'bias', 64)
                                    if biased and point in selected else zeros[256 if biased else 4])
            yield point, expected


def audit(bundle):
    import inspect
    import sdk_probe
    from csl_llm import decoder_layout, regions
    from sdk_probe import verify
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    verify(bundle)
    config = json.loads((bundle / 'config.json').read_text())
    if (config['width'], config['height'], config['weight_storage'], config['initialization']) != (
            193, 210, 'u8', 'selected-original-declarations'):
        raise ValueError('Expected the native-byte selected-original full-model candidate')
    execution = json.loads((bundle / 'execution.json').read_text())
    result = json.loads((bundle / 'results.json').read_text())
    if not execution['success'] or not result['success'] or not result['compile_only']:
        raise ValueError('Completed compilation required')
    if execution['manifest_sha256'] != digest(bundle / 'manifest.json') or execution['results_sha256'] != digest(bundle / 'results.json'):
        raise ValueError('Compilation receipt differs')
    expected = dict(expected_symbols(bundle))
    seen = set()
    images = []
    for path in sorted((bundle / 'out/bin').glob('*.elf')):
        memory = ELFMemory(str(path))
        if tuple(memory.get_fabric_dimensions()) != (200, 212):
            raise ValueError('Unexpected full-model fabric')
        with path.open('rb') as stream:
            table = ELFFile(stream).get_section_by_name('.symtab')
            if table is None:
                raise ValueError('ELF symbol table required')
            coordinates = []
            for px, py in memory.iter_coordinates():
                point = (px-4, py-1)
                if point not in expected or point in seen:
                    raise ValueError(f'Unexpected or repeated coordinate: {point}')
                seen.add(point)
                coordinates.append(list(point))
                for name, raw in expected[point].items():
                    symbols = table.get_symbol_by_name(name)
                    if not symbols or len(symbols) != 1:
                        raise ValueError(f'Expected unique symbol: {point} {name}')
                    symbol = symbols[0]
                    address, size = int(symbol['st_value']), int(symbol['st_size'])
                    if (symbol['st_info']['type'] != 'STT_OBJECT' or size != len(raw) or
                            address % 4 or address < 0 or address+size > 49152):
                        raise ValueError(f'Symbol size/alignment/range differs: {point} {name}')
                    actual = memory.read_data(px, py, address, size)
                    if len(actual) != size or any(value is None for value in actual) or bytes(actual) != raw:
                        raise ValueError(f'Actual SDK-loaded initializer differs: {point} {name}')
        images.append(dict(elf=path.name, sha256=digest(path), coordinates=coordinates))
    if seen != set(expected):
        raise ValueError('Incomplete full-model coordinate coverage')
    return dict(passed=True, application_coordinates=len(seen), images=images,
                matrix_tiles=sum(len(row['weights']) == 28672 for row in expected.values()),
                selected_matrix_tiles=config['static_input_binding']['selected_matrix_tiles'],
                initialized_controllers=config['static_input_binding']['initialized_controllers'],
                initialized_bias_roots=config['static_input_binding']['initialized_bias_roots'],
                manifest_sha256=digest(bundle / 'manifest.json'),
                execution_sha256=digest(bundle / 'execution.json'),
                auditor_sha256=digest(Path(__file__)),
                dependency_sha256={module.__name__: digest(Path(inspect.getfile(module)))
                                   for module in (sdk_probe, decoder_layout, regions)},
                scope='Actual SDK-loaded initial ELF memory only. Selected original matrices, all norms/frequencies, selected biases and zero placeholders. Original source review remains separate; no runtime identity, full original deployment or neural qualification.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    result = audit(args.bundle.resolve())
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')

"""Full-model runtime symbol contracts from actual compiler ELF tables.

Compiler-generated base-address prefixes have been observed in the pinned SDK.
Resolve exactly one actual object for each logical name; never guess an address,
prefer one ambiguous alias, or substitute initial data for stopped-core bytes.
"""
import hashlib
import inspect
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def resolve(objects, logical, size):
    pattern = re.compile(r'^(?:\$\$csl_base_address\$\$[0-9]+\$\$)?'+re.escape(logical)+r'$')
    matches = [row for row in objects if pattern.fullmatch(row['name'])]
    if len(matches) != 1:
        raise ValueError(f'Expected one actual ELF object for {logical}, found {len(matches)}')
    row = matches[0]
    alignment = 2 if logical == 'timing' else 4
    if (row['size'] != size or row['address'] < 0 or row['address'] % alignment or
            row['address']+size > 49152):
        raise ValueError(f'Invalid size, alignment or SRAM range for {logical}')
    return row


def audit(directory):
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    from csl_llm import decoder_layout, regions
    owners = {(tile.x, tile.y): tile for layer in range(24) for tile in decoder_layout.tiles(layer)}
    controllers = {tuple(decoder_layout.controller_parameters(layer)['coordinate']) for layer in range(24)}
    universe = {(x+4, y+1) for y in range(210) for x in range(193)}
    seen = set()
    records = []
    for path in sorted(Path(directory).glob('*.elf')):
        memory = ELFMemory(str(path))
        assert tuple(memory.get_fabric_dimensions()) == (200, 212)
        points = list(map(tuple, memory.iter_coordinates()))
        if not points or len(set(points)) != len(points) or seen & set(points) or not set(points) <= universe:
            raise ValueError('Invalid, duplicate or overlapping compiler coordinates')
        seen.update(points)
        with path.open('rb') as stream:
            table = ELFFile(stream).get_section_by_name('.symtab')
            if table is None:
                raise ValueError('Actual ELF symbol table is required')
            objects = [dict(name=s.name, address=int(s['st_value']), size=int(s['st_size']))
                       for s in table.iter_symbols() if s['st_info']['type'] == 'STT_OBJECT' and s['st_size']]
        symbols = {}
        logical_symbols = {}
        for px, py in points:
            x, y = px-4, py-1
            required = dict(progress=4, timing=12)
            if y < 160:
                tile = owners.get((x, y))
                matrix = tile is not None and tile.operation != 'kv'
                required.update(identity=12, status=8, weights=28672 if matrix else 4,
                    norms=3584 if (x, y) in controllers else 4,
                    frequency=128 if (x, y) in controllers else 4,
                    bias=256 if tile is not None and tile.bias_tensor else 4)
                if (x, y) in controllers:
                    required.update(input=3584, hidden=3584, output=3584)
                if tile is not None and tile.operation == 'kv':
                    required.update({'kv.keys': 8192, 'kv.values': 8192})
            else:
                matrix = x < 192 and (y-160)*192+x < 9496
                required.update(identity=4, control=8, weights=28672 if matrix else 4,
                                norms=1792 if (x, y) == (192, 160) else 4)
                if matrix and x % 8 == 0:
                    required['result'] = 512
                if (x, y) == (192, 160):
                    required.update(input=3584, lookup=3584, winner=4, best=4, summary=28)
                    required.update({'sequence.prompt': 8192, 'sequence.generated': 1024})
            for logical, size in required.items():
                row = resolve(objects, logical, size)
                actual = row['name']
                symbols[actual] = dict(address=row['address'], size=size)
                logical_symbols[logical] = actual
        records.append(dict(elf=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            coordinates=[list(point) for point in points], required_symbols=symbols,
            logical_symbols=logical_symbols))
    if seen != universe:
        raise ValueError('Incomplete full-model symbol coordinate coverage')
    return dict(passed=True, application_coordinates=len(seen), images=records,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={module.__name__: hashlib.sha256(Path(inspect.getfile(module)).read_bytes()).hexdigest()
                           for module in (decoder_layout, regions)},
        scope='Actual ELF object names/addresses/sizes/alignment for full-model stopped-core validation. '
              'Static ABI only; not original-byte, core, numerical or generation acceptance.')

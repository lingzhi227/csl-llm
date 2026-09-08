"""Check exact compiled symbol names/sizes used by full vocabulary validation."""
import argparse
import hashlib
import json
from pathlib import Path


def audit(directory):
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    seen = set()
    records = []
    matrix_count = 0
    for path in sorted(directory.glob('*.elf')):
        memory = ELFMemory(str(path))
        assert tuple(memory.get_fabric_dimensions()) == (200, 52)
        with path.open('rb') as stream:
            table = ELFFile(stream).get_section_by_name('.symtab')
            assert table is not None
            symbols = {}
            coordinates = []
            for px, py in memory.iter_coordinates():
                assert 4 <= px < 197 and 1 <= py < 51 and (px, py) not in seen
                seen.add((px, py))
                coordinates.append([px, py])
                x, y = px-4, py-1
                matrix = x < 192 and y*192+x < 9496
                controller = (x, y) == (192, 0)
                required = dict(progress=4, identity=4, control=8)
                if matrix:
                    matrix_count += 1
                    required['weights'] = 28672
                    if x % 8 == 0:
                        required['result'] = 512
                if controller:
                    required.update(lookup=3584, winner=4, best=4, timing=12)
                for name, size in required.items():
                    entries = table.get_symbol_by_name(name)
                    assert entries is not None and len(entries) == 1, (path.name, name)
                    symbol = entries[0]
                    assert symbol['st_info']['type'] == 'STT_OBJECT'
                    address = int(symbol['st_value'])
                    assert int(symbol['st_size']) == size, (path.name, name, size)
                    assert address % (2 if name == 'timing' else 4) == 0
                    assert 0 <= address < 49152 and address+size <= 49152
                    symbols[name] = dict(address=address, size=size)
        records.append(dict(elf=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                            coordinates=coordinates, required_symbols=symbols))
    assert seen == {(x+4, y+1) for y in range(50) for x in range(193)}
    assert matrix_count == 9496
    return dict(passed=True, application_coordinates=len(seen), matrix_tiles=matrix_count,
        images=records, scope='Exact symbol address/size/alignment contract for the frozen SDK core reader. Host input transfer uses the separately preserved SDK RPC metadata. Static ABI only; no numerical execution or initialized-weight acceptance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    result = audit(args.directory.resolve())
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('VOCABULARY SYMBOL CONTRACT', result['application_coordinates'], result['matrix_tiles'])

"""Original checkpoint, packed payload, exact SDK readbacks and ELF ownership."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(bundle, model):
    import numpy as np
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    config = json.loads((bundle / 'config.json').read_text())
    assert config['initialization'] == 'sdk'
    manifest = json.loads((bundle / 'manifest.json').read_text())
    for name, sha in manifest['files'].items():
        assert digest(bundle / name) == sha, name
    identity_checked = False
    if 'group_identity' in config:
        source_identity = np.load(bundle / 'identity.npy')
        actual_identity = np.load(bundle / 'actual-identity.npy')
        assert source_identity.shape == (50, 193) and source_identity.dtype == np.dtype('<u4')
        assert actual_identity.shape == (9650,) and actual_identity.dtype == np.dtype('<u4')
        expected_identity = np.zeros((50, 193), dtype='<u4')
        for y in range(50):
            for x in range(192):
                group = y*24+x//8
                if group < 1187:
                    expected_identity[y, x] = group
        np.testing.assert_array_equal(source_identity, expected_identity)
        np.testing.assert_array_equal(actual_identity.reshape(50, 193), expected_identity)
        receipt = json.loads((bundle / 'identity-load.json').read_text())
        assert receipt['exact'] is True and receipt['rectangle'] == [0, 0, 193, 50]
        assert digest(bundle / 'identity.npy') == receipt['source_sha256']
        assert digest(bundle / 'actual-identity.npy') == receipt['actual_sha256']
        identity_checked = True
    assert digest(model) == config['model_sha256']
    with model.open('rb') as stream:
        header_length = struct.unpack('<Q', stream.read(8))[0]
        header = json.loads(stream.read(header_length))
    tensor = header['model.embed_tokens.weight']
    assert tensor['dtype'] == 'BF16' and tensor['shape'] == [151936, 896]
    original = np.memmap(model, mode='r', dtype='<u2', offset=8+header_length+tensor['data_offsets'][0], shape=(151936, 896))
    packed_file = bundle / 'weights.u32.bin'
    assert packed_file.stat().st_size == 9496*7168*4
    packed = np.memmap(packed_file, mode='r', dtype='<u4', shape=(9496, 7168))
    loads = [json.loads(line) for line in (bundle / 'weight-loads.jsonl').read_text().splitlines()]
    assert len(loads) == len(config['weight_rectangles'])
    observed, records, cursor = {}, [], 0
    for block, entry in enumerate(loads):
        assert entry['block'] == block and entry['exact'] is True and entry['first_tile'] == cursor
        x, y, width, height = entry['rectangle']
        assert entry['rectangle'] == config['weight_rectangles'][block]
        assert x == 0 and cursor == y*192
        assert (y < 49 and width == 192 and 1 <= height <= 49-y) or (y == 49 and width == 88 and height == 1)
        count = width*height
        assert entry['tiles'] == count and entry['words_per_tile'] == 7168
        path = bundle / entry['actual_file']
        assert path.parent == bundle and digest(path) == entry['actual_sha256']
        actual = np.load(path, mmap_mode='r')
        assert actual.shape == (count*7168,) and actual.dtype == np.dtype('<u4')
        expected = packed[cursor:cursor+count]
        np.testing.assert_array_equal(actual.reshape(count, 7168), expected)
        assert hashlib.sha256(expected.tobytes()).hexdigest() == entry['source_bytes_sha256']
        for index in range(cursor, cursor+count):
            group, shard = divmod(index, 8)
            payload = original[group*128:(group+1)*128, shard*112:(shard+1)*112].T.copy().tobytes()
            assert packed[index].tobytes() == payload
            point = (index % 192, index//192)
            assert point not in observed
            observed[point] = hashlib.sha256(payload).hexdigest()
        records.append(dict(block=block, rectangle=entry['rectangle'], actual_sha256=entry['actual_sha256']))
        cursor += count
    assert cursor == 9496 and len(observed) == 9496
    mapped, elf_records = set(), []
    for path in sorted((bundle / 'out/bin').glob('*.elf')):
        selected = [p for x, y in ELFMemory(str(path)).iter_coordinates() if (p := (x-4, y-1)) in observed]
        if not selected:
            continue
        with path.open('rb') as stream:
            elf = ELFFile(stream)
            entries = elf.get_section_by_name('.symtab').get_symbol_by_name('weights')
            assert entries and len(entries) == 1
            symbol = entries[0]
            assert symbol['st_size'] == 28672 and isinstance(symbol['st_shndx'], int)
            section = elf.get_section(symbol['st_shndx'])
            assert section['sh_type'] in ('SHT_PROGBITS', 'SHT_NOBITS')
            offset = int(symbol['st_value'])-int(section['sh_addr'])
            assert 0 <= offset and offset+28672 <= section['sh_size']
            assert section.data()[offset:offset+28672] == bytes(28672)
            file_hash = digest(path)
            for point in selected:
                assert point not in mapped
                mapped.add(point)
                elf_records.append(dict(coordinate=point, elf=path.name, elf_sha256=file_hash,
                                        weight_address=int(symbol['st_value']), sdk_loaded_payload_sha256=observed[point]))
    assert mapped == observed.keys()
    return dict(passed=True, tiles=len(mapped), original_bytes=9496*28672, blocks=records, elf_records=elf_records,
                runtime_identity_checked=identity_checked,
                model_sha256=config['model_sha256'], packed_sha256=digest(packed_file),
                manifest_sha256=digest(bundle / 'manifest.json'), tool_sha256=digest(Path(__file__)),
                scope='Every original tied BF16 word equals packed input and full actual SDK raw readback for all9496matrixPEs; linked zero-initialized mutable weight symbol size/address and SDK coordinates verified. Does not prove model inference or dynamic stack safety.')


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
    print('EXACT SDK WEIGHT LOAD TILES', value['tiles'])

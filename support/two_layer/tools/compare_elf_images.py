"""Compare every allocated section and SDK coordinate of two frozen binaries."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def image(path):
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    with path.open('rb') as stream:
        elf = ELFFile(stream)
        sections = [dict(name=s.name, address=int(s['sh_addr']), size=int(s['sh_size']),
                         flags=int(s['sh_flags']), kind=s['sh_type'],
                         data_sha256=hashlib.sha256(s.data()).hexdigest())
                    for s in elf.iter_sections() if s['sh_flags'] & 2]
    return dict(coordinates=list(ELFMemory(str(path)).iter_coordinates()), sections=sections)


def compare(before, after):
    left = {p.name: p for p in (before / 'out/bin').glob('*.elf')}
    right = {p.name: p for p in (after / 'out/bin').glob('*.elf')}
    assert left and left.keys() == right.keys(), 'ELF name coverage differs'
    records = []
    for name in sorted(left):
        a, b = image(left[name]), image(right[name])
        assert a == b, name
        records.append(dict(name=name, before_sha256=digest(left[name]), after_sha256=digest(right[name]),
                            allocated_image=a))
    return dict(passed=True, elf_count=len(records), records=records,
                before_manifest_sha256=digest(before / 'manifest.json'),
                after_manifest_sha256=digest(after / 'manifest.json'), tool_sha256=digest(Path(__file__)),
                scope='Identical allocated section addresses/sizes/flags/types/bytes and SDK ELF coordinates for every paired ELF. Debug/symbol tables may differ. Does not prove host orchestration, runtime completion, stack safety or performance equivalence.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('before', type=Path)
    parser.add_argument('after', type=Path)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    value = compare(args.before.resolve(), args.after.resolve())
    with args.report.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print('IDENTICAL ALLOCATED ELF IMAGES', value['elf_count'])

"""Read an existing real SDK core, with original ELF symbols only as addresses.

Never constructs SdkRuntime or loads a model. Runtime output/progress must match
actual D2H and differ from the initial ELF, ruling out an initializer fallback.
"""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def audit(bundle, variant):
    import numpy as np
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    manifest = json.loads((bundle / 'manifest.json').read_text())
    execution = json.loads((bundle / 'execution.json').read_text())
    result = json.loads((bundle / 'results.json').read_text())
    assert execution['success'] and result['success']
    assert execution['manifest_sha256'] == digest(bundle / 'manifest.json')
    assert execution['results_sha256'] == digest(bundle / 'results.json')
    variant_result = next(row for row in result['variants'] if row['variant'] == variant)
    case = len(variant_result['cases'])-1
    actual_path = bundle / f'{variant}-actual-{case}.npz'
    assert digest(actual_path) == variant_result['cases'][case]['actual_sha256']
    weights_path = bundle / f'{variant}-final-weights.npy'
    assert digest(weights_path) == variant_result['final_weights_sha256']
    actual = np.load(actual_path)
    weights = np.load(weights_path)
    width = weights.shape[0]
    work = bundle / 'baseline' if variant == 'baseline' else bundle
    core_path = work / 'out.core'
    core = ELFMemory(str(core_path))
    with core_path.open('rb') as stream:
        assert ELFFile(stream)['e_type'] == 'ET_CORE'
    records = []
    seen = set()
    for path in sorted((work / 'out/bin').glob('*.elf')):
        relative = str(path.relative_to(bundle))
        assert digest(path) == manifest['files'][relative]
        initial = ELFMemory(str(path))
        assert initial.get_flags() == core.get_flags()
        assert initial.get_fabric_dimensions() == core.get_fabric_dimensions()
        with path.open('rb') as stream:
            symbols = ELFFile(stream).get_section_by_name('.symtab')
            addresses = {}
            for name in ['output', 'progress', 'weights']:
                entries = symbols.get_symbol_by_name(name)
                assert entries and len(entries) == 1
                addresses[name] = int(entries[0]['st_value']), int(entries[0]['st_size'])
        for x, y in initial.iter_coordinates():
            pe = x-4
            assert 0 <= pe < width and y == 1 and pe not in seen
            seen.add(pe)
            checks = []
            for name, expected in [('output', actual['output'][pe]),
                                   ('progress', actual['progress'][pe]), ('weights', weights[pe])]:
                address, size = addresses[name]
                assert size == expected.nbytes
                raw = core.read_data(x, y, address, size)
                assert len(raw) == size and all(value is not None for value in raw)
                payload = bytes(raw)
                assert payload == expected.tobytes(), (pe, name)
                initial_raw = bytes(initial.read_data(x, y, address, size))
                changed = payload != initial_raw
                if name in ['output', 'progress']:
                    assert changed, 'Must verify actual computed state, not initial ELF'
                checks.append(dict(symbol=name, address=address, bytes=size,
                    payload_sha256=hashlib.sha256(payload).hexdigest(), changed_from_initializer=changed))
            records.append(dict(physical_coordinate=[x, y], logical_pe=pe,
                                elf=relative, elf_sha256=digest(path), checks=checks))
    assert seen == set(range(width))
    return dict(passed=True, bundle=bundle.name, variant=variant,
        execution_sha256=digest(bundle / 'execution.json'), core_sha256=digest(core_path),
        core_bytes=core_path.stat().st_size, actual_sha256=digest(actual_path),
        final_weights_sha256=digest(weights_path), sdk_sha256=manifest['sdk_sha256'],
        records=records, auditor_sha256=digest(Path(__file__)),
        scope='Offline read of an existing SDK ET_CORE file, using pinned SDK ELFMemory ordered loader semantics and verified compiled symbol addresses. All actual D2H output/progress/weights match; output/progress differ from initial ELF. No live snapshot or SdkRuntime.read_symbol behavior, hardware transfer performance or full-model acceptance implied.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('bundle', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--variant', choices=['baseline', 'composed'], default='composed')
    args = parser.parse_args()
    value = audit(args.bundle.resolve(), args.variant)
    with args.report.open('x') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
    print('EXISTING ACTUAL SDK CORE MATCHES D2H', len(value['records']), value['core_bytes'])

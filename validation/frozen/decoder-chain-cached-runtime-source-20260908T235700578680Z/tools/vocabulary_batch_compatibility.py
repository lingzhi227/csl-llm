"""Early read-only cross-batch compatibility check, without assembling output."""
import argparse
import hashlib
import json
from pathlib import Path

from elf_partition import image, io_signature, rpc, digest
from vocabulary_partition_assemble import validate_batch


def audit(batches):
    application = {(x+4, y+1) for y in range(50) for x in range(193)}
    matrix = {(index % 192+4, index//192+1) for index in range(9496)}
    guards = application-matrix
    reference = None
    ownership = set()
    records = []
    for batch in batches:
        config, normalized, common = validate_batch(batch)
        indices = set(config['static_tile_indices'])
        assert not indices & ownership
        ownership |= indices
        wanted = {(index % 192+4, index//192+1) for index in indices}
        selected = set()
        guard_images = {}
        selected_elfs = []
        for path in sorted((batch / 'out/bin').glob('*.elf')):
            value = image(path)
            coordinates = {tuple(point) for point in value['coordinates']}
            if coordinates & wanted:
                assert coordinates <= wanted and not coordinates & selected, path.name
                selected |= coordinates
                selected_elfs.append(dict(elf=path.name, sha256=digest(path)))
            if coordinates & guards:
                assert coordinates <= guards
                for point in coordinates:
                    assert point not in guard_images
                    guard_images[point] = value
        assert selected == wanted and set(guard_images) == guards
        invariant = dict(parent_manifest=config['parent_manifest_sha256'],
            original_packed_sha256=config['source_packed_sha256'], layout=normalized,
            shared_sources=common, rpc=rpc(batch / 'out/bin/out_rpc.json'),
            metadata=json.loads((batch / 'out/out.json').read_text()),
            io=io_signature(batch / 'out'),
            guards=[dict(coordinate=list(point), image=guard_images[point]) for point in sorted(guards)])
        if reference is None:
            reference = invariant
        else:
            assert invariant == reference, f'Batch compatibility differs: {batch.name}'
        records.append(dict(bundle=batch.name, manifest_sha256=digest(batch / 'manifest.json'),
            execution_sha256=digest(batch / 'execution.json'), results_sha256=digest(batch / 'results.json'),
            resources_sha256=digest(batch / 'elf-memory.json'), dsr_sha256=digest(batch / 'dsr-graphs.json'),
            selected_indices=sorted(indices), selected_elfs=selected_elfs))
    assert records
    return dict(passed=True, original_tiles=len(ownership), guard_coordinates=len(guards),
        compatibility_sha256=hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest(),
        batches=records, auditor_sha256=digest(Path(__file__)),
        scope='Read-only existing compiled batch source/layout/RPC/I/O/ordered-loader guard equivalence and whole-ELF selected ownership. No new compilation, assembly, SdkRuntime or full vocabulary execution.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=Path, action='append', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit([path.resolve() for path in args.batch])
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('BATCHES COMPATIBLE', len(result['batches']), result['original_tiles'])

"""Whole-ELF selection and fixed SDK/program contracts for model partitions.

Reads independently audited compiler artifacts. Does not compose images, start
a simulator, qualify interlayer arithmetic or replace original-byte auditing.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def classify(images, selected, common, universe):
    if not selected or selected & common or not (selected | common) <= universe:
        raise ValueError('Invalid partition coordinate sets')
    sets = dict(selected=selected, common=common, placeholder=universe-selected-common)
    result = {key: [] for key in sets}
    seen = set()
    names = set()
    for image in images:
        points = [tuple(point) for point in image['coordinates']]
        coordinates = set(points)
        if (image['elf'] in names or not points or len(points) != len(coordinates) or
                seen & coordinates or not coordinates <= universe):
            raise ValueError('Repeated, empty or unexpected ELF ownership')
        names.add(image['elf'])
        seen |= coordinates
        categories = [name for name, region in sets.items() if coordinates <= region]
        if len(categories) != 1:
            raise ValueError('One whole ELF crosses partition boundaries; cropping is forbidden')
        result[categories[0]].append(image)
    if seen != universe:
        raise ValueError('Incomplete compiler image coverage')
    return result


def layout_without_initializers(text):
    text = re.sub(r',\.(?:weight_file|auxiliary_file)="[^"]*"', '', text)
    return re.sub(r'\.static_weights=(?:true|false)', '.static_weights=false', text)


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def audited(root):
    from sdk_probe import verify, sha
    verify(root)
    execution = json.loads((root / 'execution.json').read_text())
    result = json.loads((root / 'results.json').read_text())
    if not execution['success'] or not result['success'] or not result['offline_only']:
        raise ValueError('A completed model ELF audit is required')
    assert execution['manifest_sha256'] == sha(root / 'manifest.json')
    assert execution['results_sha256'] == sha(root / 'results.json')
    for name, expected in result['report_sha256'].items():
        assert sha(root / name) == expected
        assert json.loads((root / name).read_text())['passed']
    compiled = root / 'compiled'
    assert sha(compiled / 'execution.json') == result['source_execution_sha256']
    initializer = json.loads((root / 'initializers.json').read_text())
    assert initializer['manifest_sha256'] == sha(compiled / 'manifest.json')
    for image in initializer['images']:
        assert sha(compiled / 'out/bin' / image['elf']) == image['sha256']
    return compiled, initializer


def program_hashes(root):
    from sdk_probe import sha
    data = re.compile(r'(?:weights_\d+_\d+|bias_\d+_\d+|controller_aux_\d+|final_norm)\.csl')
    files = json.loads((root / 'manifest.json').read_text())['files']
    return {name: sha(root / name) for name in files
            if name.endswith('.csl') and name != 'layout.csl' and not data.fullmatch(name)}


def audit(candidate, baseline):
    from csl_llm.decoder_layout import tiles
    matrix = {(tile.x, tile.y) for layer in range(24) for tile in tiles(layer)
              if tile.operation != 'kv'}
    matrix |= {(index % 192, index//192+160) for index in range(9496)}
    universe = {(x, y) for y in range(210) for x in range(193)}
    assert len(matrix) == 34552 and len(universe - matrix) == 5978
    return audit_partition_geometry(candidate, baseline, matrix, universe)


def audit_partition_geometry(candidate, baseline, matrix, universe):
    """Verify intact artifacts for an explicitly declared application geometry."""
    from sdk_probe import sha
    from csl_llm import decoder_layout, regions
    from elf_partition import rpc, io_signature, image as allocated_image
    matrix, universe = set(matrix), set(universe)
    if not matrix or not matrix < universe:
        raise ValueError('Matrix ownership must be a proper subset of application coordinates')
    compiled, initial = audited(candidate)
    reference, reference_initial = audited(baseline)
    config = json.loads((compiled / 'static-input-config.json').read_text())
    selected = {tuple(row['coordinate']) for row in config['matrices']}
    assert len(selected) == len(config['matrices']) == initial['selected_matrix_tiles']
    common = universe - matrix
    assert selected <= matrix
    classes = classify(initial['images'], selected, common, universe)
    # Preserve all communication/program source and actual SDK-generated I/O.
    # Only declared initializer fields may differ in the generated layout.
    assert program_hashes(compiled) == program_hashes(reference)
    normalized = layout_without_initializers((compiled / 'layout.csl').read_text())
    assert normalized == layout_without_initializers((reference / 'layout.csl').read_text())
    out_metadata = json.loads((compiled / 'out/out.json').read_text())
    assert out_metadata == json.loads((reference / 'out/out.json').read_text()), 'Compiler output metadata differs'

    def common_signatures(root, rows):
        signatures = {}
        for row in rows:
            points = {tuple(point) for point in row['coordinates']}
            if not points & common:
                continue
            assert points <= common, 'Common image crosses matrix ownership'
            allocated = allocated_image(root / 'out/bin' / row['elf'])
            actual_coordinates = {tuple(point) for point in allocated.pop('coordinates')}
            assert actual_coordinates == {(x+4, y+1) for x, y in points}
            signature = stable_hash(allocated)
            for point in points:
                assert point not in signatures
                signatures[point] = signature
        assert set(signatures) == common
        return signatures

    common_before = common_signatures(reference, reference_initial['images'])
    common_after = common_signatures(compiled, initial['images'])
    assert common_after == common_before, 'Common allocated sections or ordered loader segments differ'
    actual_rpc, reference_rpc = rpc(compiled / 'out/bin/out_rpc.json'), rpc(reference / 'out/bin/out_rpc.json')
    actual_io, reference_io = io_signature(compiled / 'out'), io_signature(reference / 'out')
    assert actual_rpc == reference_rpc, 'SDK RPC contracts differ'
    assert actual_io == reference_io, 'Ordered SDK I/O contracts differ'
    offsets = sorted(row['original_record']['offset'] for row in config['matrices'])
    assert all(offset % 28672 == 0 for offset in offsets)
    indices = [offset//28672 for offset in offsets]
    return dict(passed=True, selected_tiles=len(selected), common_tiles=len(common),
        original_pack_indices=indices, classes=classes, program_sha256=program_hashes(compiled),
        normalized_layout_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
        output_metadata_sha256=stable_hash(out_metadata),
        common_allocated_image_by_coordinate=[dict(coordinate=list(point), signature=signature)
                                              for point, signature in sorted(common_after.items())],
        common_allocated_and_ordered_loader_segments_exact=True,
        rpc_signature_sha256=stable_hash(actual_rpc), io_signature_sha256=stable_hash(actual_io),
        candidate_audit_execution_sha256=sha(candidate / 'execution.json'),
        baseline_audit_execution_sha256=sha(baseline / 'execution.json'),
        tool_sha256=sha(Path(__file__)),
        dependency_sha256=({name: sha(Path(__file__).parent / name)
                           for name in ['sdk_probe.py', 'elf_partition.py', 'compare_elf_images.py']} |
                           {module.__name__: sha(Path(inspect.getfile(module)))
                            for module in [decoder_layout, regions]}),
        scope='Whole ELF coordinate containment and fixed program/layout/ordered SDK I/O/RPC contracts only. '
              'Selected and common images can be requested without cropping. Complete partition coverage, '
              'actual composition and SDK neural execution remain separate requirements.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('candidate', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report = audit(args.candidate.resolve(), args.baseline.resolve())
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')

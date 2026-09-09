"""Bind the existing original weight pack to current full-model ownership.

Read-only input/mapping audit. No compiler, SDK runtime or neural arithmetic.
The older pack's controller placement is not reused: only matrix coordinates
and tensor slices are compared with the current decoder/vocabulary mapping.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def check_records(records):
    from csl_llm.decoder_layout import tiles
    if len(records) != 34552:
        raise ValueError('The full original matrix pack is required')
    actual = {}
    for index, record in enumerate(records):
        point = record['x'], record['y']
        if point in actual or record['role'] != 'matrix':
            raise ValueError('Repeated or non-matrix owner')
        if record['offset'] != index*28672 or record['bytes'] != 28672:
            raise ValueError('Matrix records must cover contiguous original tile bytes')
        actual[point] = record
    expected = {}
    for layer in range(24):
        for tile in tiles(layer):
            if tile.operation != 'kv':
                expected[tile.x, tile.y] = (layer, tile.tensor, tile.row, tile.column,
                                            128, tile.valid_columns)
    for index in range(9496):
        expected[index % 192, index//192+160] = (
            -1, 'model.embed_tokens.weight', (index//8)*128, (index % 8)*112, 128, 112)
    if set(actual) != set(expected) or len(expected) != 34552:
        raise ValueError('Original matrix owners differ from the current full model')
    keys = ['layer', 'tensor', 'row', 'column', 'valid_rows', 'valid_columns']
    for point, value in expected.items():
        if tuple(actual[point][key] for key in keys) != value:
            raise ValueError(f'Original tensor slice differs at {point}')
    return len(expected)


def audit(pack, source):
    sys.path.insert(0, str(source / 'src'))
    manifest = json.loads((pack / 'manifest.json').read_text())
    assert manifest['model_sha256'] == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
    for name, expected in manifest['files'].items():
        if Path(name).name != name or digest(pack / name) != expected:
            raise ValueError(f'Changed original pack file: {name}')
    verified = json.loads((pack / 'verification.json').read_text())
    assert verified['passed'] and verified['checked_parameter_words'] == 494032768
    assert verified['manifest_sha256'] == digest(pack / 'manifest.json')
    assert verified['checker_sha256'] == digest(source / 'tools/verify_weight_pack.py')
    count = check_records(json.loads((pack / 'tile-index.json').read_text()))
    assert (pack / 'matrices.bf16').stat().st_size == count*28672
    return dict(passed=True, matrix_tiles=count, decoder_tiles=24*1044,
        vocabulary_tiles=9496, pack_manifest_sha256=digest(pack / 'manifest.json'),
        pack_verification_sha256=digest(pack / 'verification.json'),
        auditor_sha256=digest(Path(__file__)),
        source_sha256={name: digest(source / name) for name in [
            'src/csl_llm/decoder_layout.py', 'src/csl_llm/regions.py',
            'tools/verify_weight_pack.py']},
        scope='Existing hash-verified input pack and its prior raw-checkpoint verification '
              'bind every matrix coordinate/tensor slice to current full-model ownership. '
              'No old controller layout reuse, new CSL initializer, compilation, SDK loading '
              'or numerical inference acceptance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.pack.resolve(), args.source_root.resolve())
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print('ORIGINAL MODEL PACK MAPPING', result['matrix_tiles'])

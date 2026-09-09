"""Original full-model static input files, independent of CSL placement.

This emits data declarations only. It neither enables the emitters' rejected
static/u8 combination nor claims the new decoder byte ABI compiles or runs.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

from audit_model_weight_mapping import audit, digest
from compact_weights_probe import encode_initializer


def literal(name, words):
    return f'const {name}=[{len(words)}]u32'+'{'+','.join(
        f'0x{int(word):08x}' for word in words)+'};\n'


def prepare(pack, frequency_fixture, selected, source, project):
    import numpy as np
    sys.path.insert(0, str(source / 'src'))
    from csl_llm.decoder_layout import tiles, controller_parameters
    from csl_llm.decoder_storage import pairs
    from sdk_probe import verify
    mapping = audit(pack, source)
    verify(frequency_fixture)
    if (digest(frequency_fixture / 'manifest.json') !=
            'abb67d0829c8ecd2d57be77501be2c7bebd3afa35dff67c9ff15b0740b2a9828'):
        raise ValueError('The accepted pinned layer0 frequency fixture is required')
    records = json.loads((pack / 'tile-index.json').read_text())
    owners = {(row['x'], row['y']): row for row in records}
    if not 1 <= len(selected) <= 1024 or len(set(selected)) != len(selected):
        raise ValueError('Select1..1024 distinct original matrix coordinates')
    if not set(selected) <= owners.keys():
        raise ValueError('Only original matrix owners can have static weight tiles')
    # Reuse the exact frequencies from the accepted layer0 fixture. No new
    # host approximation of RoPE, and no per-step host neural computation.
    auxiliary = np.load(pack / 'auxiliary.npz')
    words = np.array([int(x, 16) for x in re.findall(r'0x([0-9a-fA-F]{8})',
        (frequency_fixture / 'controller_aux.csl').read_text())], np.uint32)
    assert len(words) == 928
    first_norms = pairs(np.concatenate([auxiliary[f'model.layers.0.{name}.weight']
        for name in ['input_layernorm', 'post_attention_layernorm']]))
    np.testing.assert_array_equal(words[:896], first_norms)
    frequency = words[896:]
    assert np.isfinite(frequency.view(np.float32)).all()
    frequency_text = 'const frequency=[32]f32{'+','.join(
        f'@bitcast(f32,@as(u32,0x{int(word):08x}))' for word in frequency)+'};\n'
    out = project / 'evidence' / ('model-static-inputs-'+datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    matrix_files = []
    with (pack / 'matrices.bf16').open('rb') as stream:
        for point in sorted(selected, key=lambda p: (p[1], p[0])):
            row = owners[point]
            stream.seek(row['offset'])
            payload = stream.read(28672)
            assert len(payload) == 28672
            name = f'weights_{point[0]}_{point[1]}.csl'
            (out / name).write_text(encode_initializer(payload, 'bytes'))
            matrix_files.append(dict(coordinate=list(point), file=name,
                original_record=row, original_bytes_sha256=hashlib.sha256(payload).hexdigest()))
    auxiliary_files = []
    for layer in range(24):
        controller = controller_parameters(layer)
        point = controller['coordinate']
        norms = pairs(np.concatenate([auxiliary[name] for name in controller['norms']]))
        assert len(norms) == 896
        name = f'controller_aux_{layer}.csl'
        (out / name).write_text(literal('norms', norms)+frequency_text)
        auxiliary_files.append(dict(coordinate=list(point), file=name,
            tensors=list(controller['norms']), kind='decoder_norms_frequency'))
        for tile in tiles(layer):
            if not tile.bias_tensor:
                continue
            bias = pairs(auxiliary[tile.bias_tensor][tile.row:tile.row+128])
            assert len(bias) == 64
            name = f'bias_{tile.x}_{tile.y}.csl'
            (out / name).write_text(literal('bias', bias))
            auxiliary_files.append(dict(coordinate=[tile.x, tile.y], file=name,
                tensor=tile.bias_tensor, row=tile.row, kind='bias'))
    final_norm = pairs(auxiliary['model.norm.weight'])
    assert len(final_norm) == 448
    (out / 'final_norm.csl').write_text(literal('norms', final_norm))
    auxiliary_files.append(dict(coordinate=[192, 160], file='final_norm.csl',
        tensor='model.norm.weight', kind='final_norm'))
    assert len(auxiliary_files) == 241
    auxiliary.close()
    config = dict(original_pack=pack.name, mapping_audit=mapping,
        frequency_fixture=frequency_fixture.name,
        frequency_fixture_manifest_sha256=digest(frequency_fixture / 'manifest.json'),
        frequency_auxiliary_sha256=digest(frequency_fixture / 'controller_aux.csl'),
        matrices=matrix_files, auxiliaries=auxiliary_files,
        scope='Selected original BF16 matrix byte declarations and all24layer norms/biases/'
              'pinnedfrequencies plus final norm. Data authoring only: no placement binding, '
              'new ELF/alignment, SDK loading or model inference acceptance.')
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    for name in ['model_static_inputs.py', 'audit_model_weight_mapping.py',
                 'compact_weights_probe.py', 'verify_weight_pack.py', 'sdk_probe.py']:
        shutil.copy2(source / 'tools' / name, out / name)
    for name in ['decoder_storage.py', 'decoder_layout.py', 'regions.py']:
        shutil.copy2(source / 'src/csl_llm' / name, out / name)
    (out / 'manifest.json').write_text(json.dumps(dict(
        files={p.name: digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pack', type=Path, required=True)
    parser.add_argument('--frequency-fixture', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--tile', type=lambda text: tuple(map(int, text.split(','))), action='append', required=True)
    args = parser.parse_args()
    prepare(args.pack.resolve(), args.frequency_fixture.resolve(), args.tile,
            args.source_root.resolve(), args.project_root.resolve())

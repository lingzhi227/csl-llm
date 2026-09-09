"""Original checkpoint -> mutable transfer packer -> frozen literal equivalence.

This is an offline initializer audit, not actual SDK loading or numerical output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def literals(path):
    return np.array([int(word, 16) for word in re.findall(r'0x([0-9a-fA-F]{8})', path.read_text())], np.uint32)


def audit(source, project, fixtures):
    sys.path.insert(0, str(source / 'src'))
    from csl_llm.decoder_layout import tiles
    from csl_llm.decoder_storage import layer_transfers
    from safetensors import safe_open
    import torch
    torch.set_num_threads(1)
    model = project / 'models/qwen2.5-0.5b-7ae5576/model.safetensors'
    model_sha = digest(model)
    if model_sha != 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe':
        raise ValueError('Wrong original checkpoint')
    results = []
    with safe_open(model, framework='pt', device='cpu') as checkpoint:
        for fixture in fixtures:
            manifest = json.loads((fixture / 'manifest.json').read_text())
            for name, sha in manifest['files'].items():
                if Path(name).name != name or digest(fixture / name) != sha:
                    raise ValueError(f'Changed frozen fixture {name}')
            config = json.loads((fixture / 'config.json').read_text())
            layer = config.get('layer', 0)
            names = {t.tensor for t in tiles(layer) if t.tensor}
            names |= {t.bias_tensor for t in tiles(layer) if t.bias_tensor}
            names |= {f'model.layers.{layer}.{name}.weight' for name in ['input_layernorm', 'post_attention_layernorm']}
            tensors = {}
            for name in names:
                value = checkpoint.get_tensor(name)
                if value.dtype != torch.bfloat16:
                    raise ValueError(f'Unexpected original storage {name}')
                tensors[name] = value.view(torch.uint16).numpy()
            auxiliary = literals(fixture / 'controller_aux.csl')
            assert len(auxiliary) == 928
            frequency = auxiliary[896:].copy().view(np.float32)
            expected_identity = np.zeros((20, 64, 3), np.uint32)
            expected_bias = {}
            for line in (fixture / 'layout.csl').read_text().splitlines():
                if not line.startswith('@set_tile_code'):
                    continue
                x, y = map(int, re.match(r'@set_tile_code\((\d+),(\d+),', line).groups())
                values = [int(re.search(r'\.'+key+r'=(\d+)', line)[1]) for key in ['group', 'destination_x', 'destination_y']]
                expected_identity[y, x] = values
                aux_name = re.search(r'\.auxiliary_file="([^"]+)"', line)[1]
                if aux_name.startswith('bias_'):
                    expected_bias[x, y] = literals(fixture / aux_name)
            count = 0
            transfer_hashes = []
            for transfer in layer_transfers(tensors, frequency, layer, origin=(0, 0)):
                values = transfer.values
                if transfer.symbol == 'identity':
                    np.testing.assert_array_equal(values, expected_identity)
                elif transfer.symbol == 'weights':
                    for y in range(transfer.height):
                        for x in range(transfer.width):
                            baseline = fixture / f'weights_{transfer.x+x}_{transfer.y+y}.csl'
                            np.testing.assert_array_equal(values[y, x], literals(baseline))
                            count += 1
                elif transfer.symbol == 'bias':
                    assert len(expected_bias) == transfer.width*transfer.height == 9
                    for y in range(transfer.height):
                        np.testing.assert_array_equal(values[y, 0], expected_bias[transfer.x, transfer.y+y])
                elif transfer.symbol == 'norms':
                    np.testing.assert_array_equal(values.ravel(), auxiliary[:896])
                else:
                    assert transfer.symbol == 'frequency'
                    np.testing.assert_array_equal(values.ravel(), auxiliary[896:])
                transfer_hashes.append(dict(symbol=transfer.symbol,
                    rectangle=[transfer.x, transfer.y, transfer.width, transfer.height],
                    words_per_pe=transfer.words_per_pe, bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest()))
            assert count == 1044
            results.append(dict(layer=layer, fixture=fixture.name, fixture_manifest_sha256=digest(fixture / 'manifest.json'),
                matrix_tiles=count, identity_pes=1280, bias_roots=9, exact=True, transfers=transfer_hashes))
    return dict(passed=True, model_sha256=model_sha, layers=results,
        sources={str(path.relative_to(source)):digest(path) for path in [
            source / 'src/csl_llm/decoder_layout.py', source / 'src/csl_llm/decoder_storage.py', source / 'src/csl_llm/regions.py']},
        auditor_sha256=digest(Path(__file__)),
        scope='Offline exact original BF16 transfer packing and equivalence to frozen selected-layer literals/coordinates. Frequencies reused from frozen fixture. No SDK initialization, dynamic memory, streamed decoder arithmetic or24layer inference acceptance.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--fixture', type=Path, action='append', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source_root.resolve(), args.project_root.resolve(), [p.resolve() for p in args.fixture])
    with args.report.open('x') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    print('OFFLINE ORIGINAL DECODER TRANSFERS PASS', [row['layer'] for row in result['layers']])

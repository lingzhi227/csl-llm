"""Independent original-input two-layer CPU fixture, never a device compute path."""
import argparse
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
ROOT = Path(__file__).resolve().parents[1]


def metric(actual, expected):
    import numpy as np
    actual, expected = np.asarray(actual, dtype=np.float64), np.asarray(expected, dtype=np.float64)
    if actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError('Reference arrays must have identical shapes and finite values')
    difference = actual-expected
    absolute = float(np.max(np.abs(difference)))
    norm = float(np.linalg.norm(difference)/max(float(np.linalg.norm(expected)), 1e-30))
    peak = absolute/max(float(np.max(np.abs(expected))), 1e-30)
    if not (absolute <= 1e-5 or (norm <= 2e-5 and peak <= 3e-5)):
        raise ValueError(f'Independent CPU/official reference disagreement: {absolute}, {norm}, {peak}')
    return dict(max_abs=absolute, relative_l2=norm, relative_peak=peak)


def prepare(project, source, prefix_length=1):
    if type(prefix_length) is not int or prefix_length not in (1, 2):
        raise ValueError("Only the original one- or two-token diagnostic prefix is supported")
    import numpy as np
    import torch
    from safetensors import safe_open
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import verify, sha
    from resident_decoder_probe import explicit_layer
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = project / 'models/qwen2.5-0.5b-7ae5576'
    reference = project / 'evidence/reference-f32-greedy-20260908T0554'
    fixture = project / 'evidence/resident-decoder-20260908T113615465376Z'
    verify(fixture)
    assert sha(fixture / 'manifest.json') == 'abb67d0829c8ecd2d57be77501be2c7bebd3afa35dff67c9ff15b0740b2a9828'
    model_sha = sha(model / 'model.safetensors')
    assert model_sha == 'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'
    fixture_config = json.loads((fixture / 'config.json').read_text())
    assert fixture_config['model_sha256'] == model_sha
    assert fixture_config['reference_report_sha256'] == sha(reference / 'report.json')
    report = json.loads((reference / 'report.json').read_text())
    assert report['model_manifest_sha256'] == sha(model / 'manifest.json')
    assert report['success'] and report['dtype'] == 'float32'
    trace = reference / 'arithmetic/step-000.npz'
    assert sha(trace) == report['files']['arithmetic/step-000.npz']
    tokens = next(row for row in report['prompts'] if row['id'] == 'arithmetic')['prompt_tokens'][:prefix_length]
    assert len(tokens) == prefix_length
    token = tokens[0]
    official = np.load(trace)
    original_fixture = np.load(fixture / 'inputs.npz')
    with safe_open(model / 'model.safetensors', framework='pt', device='cpu') as handle:
        state = {name: handle.get_tensor(name) for name in handle.keys()
                 if name.startswith(('model.layers.0.', 'model.layers.1.'))}
        assert len(state) == 24
        hidden = torch.cat([handle.get_slice('model.embed_tokens.weight')[value:value+1].float() for value in tokens])
    np.testing.assert_array_equal(hidden.numpy(), original_fixture['inputs'][:prefix_length])
    np.testing.assert_array_equal(hidden.numpy(), official['model.embed_tokens'][0, :prefix_length])
    accepted_chain = None
    if prefix_length == 2:
        accepted = project / 'evidence/decoder-chain-reference-20260908T222936118540Z'
        verify(accepted)
        assert sha(accepted / 'manifest.json') == '9c2eb93b26f396e71f8ca43b5501d0ea8408f8c2595f5818456332b1a1f19ab5'
        accepted_chain = np.load(accepted / 'inputs.npz')
        np.testing.assert_array_equal(hidden.numpy()[:1], accepted_chain['input'])
    arrays = {'input': hidden.numpy().copy()}
    checks = []
    mapping = dict(input_norm='input_layernorm', q='self_attn.q_proj', k='self_attn.k_proj',
                   v='self_attn.v_proj', o='self_attn.o_proj', post_norm='post_attention_layernorm',
                   up='mlp.up_proj', gate='mlp.gate_proj', down='mlp.down_proj')
    frequency_bits = np.array([int(word, 16) for word in __import__('re').findall(
        r'0x([0-9a-fA-F]{8})', (fixture / 'controller_aux.csl').read_text())][-32:], np.uint32)
    with torch.inference_mode():
        for layer in range(2):
            nodes, keys, values, frequency = explicit_layer(state, hidden, layer)
            np.testing.assert_array_equal(frequency.numpy().view(np.uint32), frequency_bits)
            prefix = f'model.layers.{layer}'
            for position in range(prefix_length):
                for name, suffix in mapping.items():
                    expected = official[prefix+'.'+suffix][0, position:position+1]
                    checks.append(dict(layer=layer, position=position, stage=name,
                                       **metric(nodes[name].numpy()[position:position+1], expected)))
                checks.append(dict(layer=layer, position=position, stage='output', **metric(
                    nodes['output'].numpy()[position:position+1], official[prefix][0, position:position+1])))
                for name, cache_tensor in [('cache_k', keys), ('cache_v', values)]:
                    suffix = 'key' if name == 'cache_k' else 'value'
                    checks.append(dict(layer=layer, position=position, stage=name, **metric(
                        cache_tensor.numpy()[:, position:position+1], official[f'cache.{layer}.{suffix}'][0, :, position:position+1])))
            if accepted_chain is not None:
                for name, value in nodes.items():
                    checks.append(dict(layer=layer, position=0, stage=name, reference='accepted single-token',
                                       **metric(value.numpy()[:1], accepted_chain[f'layer_{layer}_{name}'])))
            for name, value in nodes.items():
                arrays[f'layer_{layer}_{name}'] = value.numpy().copy()
            arrays[f'layer_{layer}_cache_k'] = keys.numpy().copy()
            arrays[f'layer_{layer}_cache_v'] = values.numpy().copy()
            # Primary layer-1 oracle consumes independent layer-0 output.
            # It never consumes an SDK result or a replaced official boundary.
            hidden = nodes['output']
    assert all(np.isfinite(value).all() for value in arrays.values())
    label = 'decoder-chain-cached-reference-' if prefix_length == 2 else 'decoder-chain-reference-'
    out = project / 'evidence' / (label + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    np.savez(out / 'inputs.npz', **arrays)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/resident_decoder_probe.py', out / 'equations.py')
    shutil.copy2(source / 'configs/precision.json', out / 'precision.json')
    (out / 'config.json').write_text(json.dumps(dict(token_id=token, token_ids=tokens, prefix_length=prefix_length, positions=list(range(prefix_length)), position=0, layers=[0, 1],
        model_sha256=model_sha, model_manifest_sha256=sha(model / 'manifest.json'),
        official_report_sha256=sha(reference / 'report.json'), official_trace_sha256=sha(trace),
        accepted_layer0_manifest_sha256=sha(fixture / 'manifest.json'),
        frequency_auxiliary_sha256=sha(fixture / 'controller_aux.csl'),
        cpu_checks=checks, threads=1, python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'safetensors')},
        scope='Independent original-input CPU equations for complete layers0→1, cross-checked '
              f'against official FP32 causal prefix of {prefix_length} tokens. Fixture only, not SDK execution or full-model logits.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(files={p.name: sha(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--prefix-length', type=int, choices=[1, 2], default=1)
    args = parser.parse_args()
    prepare(args.project_root.resolve(), args.source_root.resolve(), args.prefix_length)

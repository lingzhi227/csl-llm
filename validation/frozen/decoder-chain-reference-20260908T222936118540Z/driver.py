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


def prepare(project, source):
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
    token = next(row for row in report['prompts'] if row['id'] == 'arithmetic')['prompt_tokens'][0]
    official = np.load(trace)
    original_fixture = np.load(fixture / 'inputs.npz')
    with safe_open(model / 'model.safetensors', framework='pt', device='cpu') as handle:
        state = {name: handle.get_tensor(name) for name in handle.keys()
                 if name.startswith(('model.layers.0.', 'model.layers.1.'))}
        assert len(state) == 24
        hidden = handle.get_slice('model.embed_tokens.weight')[token:token+1].float()
    np.testing.assert_array_equal(hidden.numpy()[0], original_fixture['inputs'][0])
    np.testing.assert_array_equal(hidden.numpy()[0], official['model.embed_tokens'][0, 0])
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
            for name, suffix in mapping.items():
                expected = official[prefix+'.'+suffix][0, :1]
                checks.append(dict(layer=layer, stage=name, **metric(nodes[name].numpy(), expected)))
            checks.append(dict(layer=layer, stage='output', **metric(nodes['output'].numpy(), official[prefix][0, :1])))
            checks.append(dict(layer=layer, stage='cache_k', **metric(keys.numpy(), official[f'cache.{layer}.key'][0, :, :1])))
            checks.append(dict(layer=layer, stage='cache_v', **metric(values.numpy(), official[f'cache.{layer}.value'][0, :, :1])))
            for name, value in nodes.items():
                arrays[f'layer_{layer}_{name}'] = value.numpy().copy()
            arrays[f'layer_{layer}_cache_k'] = keys.numpy().copy()
            arrays[f'layer_{layer}_cache_v'] = values.numpy().copy()
            # Primary layer-1 oracle consumes independent layer-0 output.
            # It never consumes an SDK result or a replaced official boundary.
            hidden = nodes['output']
    assert all(np.isfinite(value).all() for value in arrays.values())
    out = project / 'evidence' / ('decoder-chain-reference-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    np.savez(out / 'inputs.npz', **arrays)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/resident_decoder_probe.py', out / 'equations.py')
    shutil.copy2(source / 'configs/precision.json', out / 'precision.json')
    (out / 'config.json').write_text(json.dumps(dict(token_id=token, position=0, layers=[0, 1],
        model_sha256=model_sha, model_manifest_sha256=sha(model / 'manifest.json'),
        official_report_sha256=sha(reference / 'report.json'), official_trace_sha256=sha(trace),
        accepted_layer0_manifest_sha256=sha(fixture / 'manifest.json'),
        frequency_auxiliary_sha256=sha(fixture / 'controller_aux.csl'),
        cpu_checks=checks, threads=1, python=sys.version,
        versions={name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'safetensors')},
        scope='Independent original-input CPU equations for complete layers0→1, cross-checked '
              'against official FP32 causal position0. Fixture only, not SDK execution or full-model logits.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(files={p.name: sha(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    prepare(args.project_root.resolve(), args.source_root.resolve())

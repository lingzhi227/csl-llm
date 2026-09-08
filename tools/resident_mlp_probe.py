"""Native resident MLP candidate; host transfers only input and final diagnostics.

No intermediate host operation is part of the device inference path. Independent
CPU references are prepared before the immutable SDK bundle is executed.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
WIDTH, HEIGHT = 64, 20
CONTROLLER = (62, 0)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def geometry():
    sys.path.insert(0, str(ROOT / 'src'))
    from csl_llm.resident_mlp_regions import Region
    return {
        'up': [Region(8 * (g % 2), g // 2, 8, 1) for g in range(38)],
        'gate': [Region(16 + 8 * (g % 2), g // 2, 8, 1) for g in range(38)],
        'down': [Region(32, 2 * g, 22, 2) for g in range(7)],
    }


def make_layout(regions, weight_files=None):
    weight_files = weight_files or {}
    occupied = {}
    for role, name in enumerate(('up', 'gate', 'down'), 1):
        for group, region in enumerate(regions[name]):
            for ordinal in range(region.participants):
                point = region.point(ordinal)
                assert point not in occupied
                destination = regions['gate'][group].point(0) if name == 'up' else CONTROLLER
                occupied[point] = (role, group, ordinal, region.participants,
                                   *region.neighbors(ordinal), *destination)
    lines = ['const memcpy=@import_module("<memcpy/get_params>",.{.width=64,.height=20});',
             'layout{@set_rectangle(64,20);']
    for y in range(HEIGHT):
        for x in range(WIDTH):
            values = occupied.get((x, y), (4 if (x, y) == CONTROLLER else 0,
                                          0, 0, 2, 'WEST', 'EAST', 62, 0))
            role, group, ordinal, participants, negative, positive, dx, dy = values
            # Only output roots consume group/destination. Coalescing the other
            # roles avoids recompiling identical numeric code per weight tile.
            if role in (1, 2, 3) and ordinal != 0:
                group, dx, dy = 0, 62, 0
                if role == 2:
                    role = 1
            params = (f'.memcpy_params=memcpy.get_params({x}),.role={role},.group={group},'
                      f'.ordinal={ordinal},.participants={participants},.negative={negative},'
                      f'.positive={positive},.destination_x={dx},.destination_y={dy}')
            if (x, y) in weight_files:
                params += f',.static_weights=true,.weight_file="{weight_files[x, y]}"'
            lines.append(f'@set_tile_code({x},{y},"pe.csl",.{{{params}}});')
            for color, left, right, last_y, period in [(3, 0, 31, 18, 896), (4, 32, 53, 13, 4928)]:
                routes = None
                filtered = False
                if (x, y) == CONTROLLER:
                    routes = '.rx=.{RAMP},.tx=.{WEST}'
                elif y == 0 and left <= x < 62:
                    outputs = (['WEST'] if x > left else [])
                    if x <= right:
                        outputs += ['SOUTH', 'RAMP']
                        filtered = True
                    routes = '.rx=.{EAST},.tx=.{' + ','.join(outputs) + '}'
                elif 0 < y <= last_y and left <= x <= right:
                    routes = '.rx=.{NORTH},.tx=.{RAMP' + (',SOUTH' if y < last_y else '') + '}'
                    filtered = True
                if routes is None:
                    continue
                filter_text = ''
                if filtered:
                    shard = x % 8 if color == 3 else (y % 2) * 22 + x - 32
                    counter = (period - shard * 112) % period
                    filter_text = (f',.filter=.{{.kind=.{{.counter=true}},.count_data=true,'
                                   f'.init_counter={counter},.max_counter=111,.limit1={period-1}}}')
                lines.append(f'@set_color_config({x},{y},@get_color({color}),.{{.routes=.{{{routes}}}{filter_text}}});')
    for name, kind in [('weights', 'u32'), ('input', 'f32'), ('result', 'f32'),
                       ('activation', 'f32'), ('output', 'f32'), ('progress', 'u32'), ('timing', 'u16')]:
        lines.append(f'@export_name("{name}",[*]{kind},false);')
    for name in ['initialize', 'prepare', 'compute']:
        lines.append(f'@export_name("{name}",fn()void);')
    return '\n'.join(lines + ['}']) + '\n'


def copy_sources(out):
    copies = [(Path(__file__), 'driver.py'), (ROOT / 'tools/sdk_probe.py', 'executor.py'),
              (ROOT / 'tools/bf16_transport.py', 'bf16_transport.py'),
              (ROOT / 'csl/programs/resident_mlp/pe.csl', 'pe.csl'),
              (ROOT / 'csl/programs/resident_mlp/kernel.csl', 'kernel.csl'),
              (ROOT / 'csl/programs/resident_mlp/vector.csl', 'vector.csl'),
              (ROOT / 'csl/programs/resident_mlp/line.csl', 'line.csl'),
              (ROOT / 'csl/programs/resident_mlp/input.csl', 'input.csl'),
              (ROOT / 'csl/programs/resident_mlp/packets.csl', 'packets.csl'),
              (ROOT / 'src/csl_llm/resident_mlp_regions.py', 'regions.py'),
              (ROOT / 'configs/precision.json', 'precision.json')]
    for source, destination in copies:
        shutil.copy2(source, out / destination)


def write_manifest(out):
    (out / 'manifest.json').write_text(json.dumps(dict(
        sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
        files={p.name: digest(p) for p in out.iterdir() if p.is_file()}), indent=2) + '\n')


def prepare_static(source):
    """Reuse frozen validation fixtures, compile each original tile into CSL data."""
    import numpy as np
    sys.path.insert(0, str(ROOT / 'tools'))
    from bf16_transport import pack_bf16_pairs
    manifest = json.loads((source / 'manifest.json').read_text())
    for name in ('inputs.npz', 'config.json'):
        assert digest(source / name) == manifest['files'][name]
    out = ROOT / 'evidence' / ('resident-mlp-static-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    shutil.copy2(source / 'inputs.npz', out / 'inputs.npz')
    copy_sources(out)
    data = np.load(out / 'inputs.npz')
    weight_files = {}
    for name, x_offset in [('up', 0), ('gate', 16), ('down', 32)]:
        tiles = data['weights_' + name]
        for y in range(tiles.shape[0]):
            for x in range(tiles.shape[1]):
                words = pack_bf16_pairs(tiles[y, x])
                filename = f'weights_{x+x_offset}_{y}.csl'
                (out / filename).write_text('const values=[7168]u32{' + ','.join(f'0x{int(word):08x}' for word in words) + '};\n')
                weight_files[x+x_offset, y] = filename
    assert len(weight_files) == 916
    (out / 'layout.csl').write_text(make_layout(geometry(), weight_files))
    config = json.loads((source / 'config.json').read_text())
    config.update(static_weights=True, initializer_tiles=916,
                  initializer_payload_bytes=916*7168*4,
                  source_fixture_bundle=source.name,
                  source_fixture_manifest_sha256=digest(source / 'manifest.json'),
                  weight_initialization='Ordinary CSL global arrays through cslc and SDK load/run; no runtime weight H2D, no debugger writes or ELF patches.')
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    write_manifest(out)
    print(out, flush=True)


def prepare():
    import numpy as np
    import torch
    from safetensors.torch import load_file
    out = ROOT / 'evidence' / ('resident-mlp-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    model = ROOT / 'models/qwen2.5-0.5b-7ae5576/model.safetensors'
    trace = ROOT / 'evidence/reference-f32/arithmetic/step-000.npz'
    state, ref = load_file(model), np.load(trace)
    x = ref['model.layers.0.post_attention_layernorm'][0, -1].copy()
    inputs = np.stack([x, -x, np.zeros_like(x), x])
    regions = geometry()
    data = {'inputs': inputs, 'official': ref['model.layers.0.mlp.down_proj'][0, -1]}
    tensors = {}
    for name in regions:
        tensor = f'model.layers.0.mlp.{name}_proj.weight'
        weight = state[tensor].float()
        tensors[name] = weight
        width, height = (22, 14) if name == 'down' else (16, 19)
        packed = np.zeros((height, width, 112, 128), np.uint16)
        x_offset = {'up': 0, 'gate': 16, 'down': 32}[name]
        for group, region in enumerate(regions[name]):
            for ordinal in range(region.participants):
                px, py = region.point(ordinal)
                shard = ordinal if name != 'down' or ordinal < 22 else 65 - ordinal
                valid = min(112, weight.shape[1] - shard * 112)
                tile = weight[group * 128:(group + 1) * 128, shard * 112:shard * 112 + valid].T.contiguous().numpy()
                packed[py, px - x_offset, :valid] = (tile.view(np.uint32) >> 16).astype(np.uint16)
        data['weights_' + name] = packed
    # These tensors are validation fixtures only; the worker never transfers them.
    with torch.no_grad():
        up = torch.from_numpy(inputs) @ tensors['up'].T
        gate = torch.from_numpy(inputs) @ tensors['gate'].T
        activation = torch.nn.functional.silu(gate) * up
        output = activation @ tensors['down'].T
    for name, value in [('up', up), ('gate', gate), ('activation', activation), ('output', output)]:
        data['expected_' + name] = value.numpy()
    np.savez(out / 'inputs.npz', **data)
    copy_sources(out)
    (out / 'layout.csl').write_text(make_layout(regions))
    config = dict(model_sha256=digest(model), trace_sha256=digest(trace),
                  cases=['canonical', 'negative', 'zero', 'canonical-repeat'],
                  roots={name: [list(r.point(0)) for r in rs] for name, rs in regions.items()},
                  scope='Full native layer0 resident UP/GATE/SwiGLU/DOWN, device-only intermediate computation and transfer; not RMS/residual, decoder layer or model.')
    (out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    write_manifest(out)
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from bf16_transport import pack_bf16_pairs
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    verify(root)
    os.chdir(root)
    config = json.loads((root / 'config.json').read_text())
    policy = json.loads((root / 'precision.json').read_text())['stage_gate']['absolute_or_normwise']
    data = np.load(root / 'inputs.npz')
    worker_started = time.monotonic()

    def stage(value):
        event = dict(stage=value, elapsed_seconds=time.monotonic()-worker_started)
        (root / 'stage.json').write_text(json.dumps(event) + '\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event) + '\n')
        print(value, flush=True)

    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=71,22',
               '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1', '--dump-dsr-alloc-graph']
    (root / 'compile-command.json').write_text(json.dumps(command) + '\n')
    stage('compile')
    subprocess.run(command, check=True)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True, num_threads=8, dump_core=True), SdkTarget.WSE3))
    ids = {name: runner.get_id(name) for name in ['weights', 'input', 'result', 'activation', 'output', 'progress', 'timing']}
    stage('sdk-load')
    runner.load()
    stage('sdk-run')
    runner.run()

    def read(name, x, y, width, height, count):
        value = np.zeros(width * height * count, np.uint32 if name in ('progress', 'timing') else np.float32)
        runner.memcpy_d2h(value, ids[name], x, y, width, height, count, streaming=False,
                          data_type=MemcpyDataType.MEMCPY_16BIT if name == 'timing' else MemcpyDataType.MEMCPY_32BIT,
                          order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        return value

    def compare(actual, expected):
        assert actual.shape == expected.shape and np.isfinite(actual).all()
        delta = actual.astype(np.float64) - expected.astype(np.float64)
        absolute = float(np.max(np.abs(delta)))
        norm = float(np.linalg.norm(delta) / max(np.linalg.norm(expected.astype(np.float64)), 1e-30))
        peak = float(absolute / max(np.max(np.abs(expected)), 1e-30))
        assert absolute <= policy['max_abs'] or (norm <= policy['both']['relative_l2'] and peak <= policy['both']['relative_peak']), (absolute, norm, peak)
        return dict(max_abs=absolute, relative_l2=norm, relative_peak=peak)

    reports = []
    try:
        stage('weights-in-ELF' if config.get('static_weights') else 'load-resident-weights')
        if not config.get('static_weights'):
            for name, x, width, height in [('up', 0, 16, 19), ('gate', 16, 16, 19), ('down', 32, 22, 14)]:
                runner.memcpy_h2d(ids['weights'], pack_bf16_pairs(data['weights_' + name]), x, 0, width, height,
                                  7168, streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                                  order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        stage('initialize')
        runner.launch('initialize', nonblock=False)
        for epoch, case in enumerate(config['cases']):
            stage(case)
            runner.memcpy_h2d(ids['input'], data['inputs'][epoch].copy(), 62, 0, 1, 1, 896,
                              streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                              order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            stage(case + '-prepare-launch')
            runner.launch('prepare', nonblock=False)
            stage(case + '-read-ready')
            ready = read('progress', 0, 0, WIDTH, HEIGHT, 1)
            np.save(root / f'ready-{epoch}.npy', ready)
            np.testing.assert_array_equal(ready, np.full(WIDTH * HEIGHT, 2 * epoch + 1, np.uint32))
            stage(case + '-compute-and-read-output')
            runner.launch('compute', nonblock=False)
            stage(case + '-read-output')
            # This controller command stays blocked until the entire MLP finishes.
            actual = {'output': read('output', 62, 0, 1, 1, 896)}
            np.save(root / f'output-{epoch}.npy', actual['output'])
            stage(case + '-diagnostics')
            actual['activation'] = read('activation', 62, 0, 1, 1, 4928)
            np.save(root / f'activation-{epoch}.npy', actual['activation'])
            actual['progress'] = read('progress', 0, 0, WIDTH, HEIGHT, 1)
            np.save(root / f'progress-{epoch}.npy', actual['progress'])
            actual['timing'] = read('timing', 62, 0, 1, 1, 6)
            actual['ready'] = ready
            for name in ['up', 'gate', 'down']:
                height = 14 if name == 'down' else 19
                columns = {}
                for x in sorted({p[0] for p in config['roots'][name]}):
                    value = read('result', x, 0, 1, height, 128).reshape(height, 128)
                    actual[f'{name}_column_{x}'] = value
                    columns[x] = value
                actual[name] = np.concatenate([columns[x][y] for x, y in config['roots'][name]])
            filename = root / f'actual-{epoch}.npz'
            np.savez(filename, **actual)
            np.testing.assert_array_equal(actual['progress'], np.full(WIDTH * HEIGHT, 2 * epoch + 2, np.uint32))
            assert np.all(actual['activation'][4864:].view(np.uint32) == 0)
            np.testing.assert_array_equal(actual['output'].view(np.uint32), actual['down'].view(np.uint32))
            checks = {name: compare(actual[name][:4864] if name == 'activation' else actual[name], data['expected_' + name][epoch])
                      for name in ['up', 'gate', 'activation', 'output']}
            if epoch in (0, 3):
                checks['official'] = compare(actual['output'], data['official'])
            if epoch == 2:
                assert np.all(actual['output'] == 0)
            if epoch == 3:
                np.testing.assert_array_equal(actual['output'].view(np.uint32), np.load(root / 'actual-0.npz')['output'].view(np.uint32))
            t = actual['timing'].astype(np.uint64)
            assert np.all(t < 65536)
            ticks = lambda a: int(a[0]) + (int(a[1]) << 16) + (int(a[2]) << 32)
            cycles = (ticks(t[3:]) - ticks(t[:3])) & ((1 << 48) - 1)
            assert 0 < cycles < (1 << 40)
            reports.append(dict(case=case, checks=checks, controller_cycles=cycles, actual_sha256=digest(filename)))
            (root / 'results.json').write_text(json.dumps(dict(success=False, cases=reports), indent=2) + '\n')
    finally:
        runner.stop()
    (root / 'results.json').write_text(json.dumps(dict(success=True, cases=reports, scope=config['scope']), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--prepare-static-from', type=Path)
    parser.add_argument('--worker', type=Path)
    parser.add_argument('--execute', type=Path)
    args = parser.parse_args()
    if args.prepare_static_from:
        prepare_static(args.prepare_static_from.resolve())
    elif args.prepare:
        prepare()
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 1800)

"""Full tied vocabulary candidate; host computes references only in preparation."""
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
GROUPS, WIDTH, HEIGHT = 1187, 193, 50
CONTROLLER = (192, 0)
SDK_SHA = 'fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def point(group, shard):
    if not 0 <= group < GROUPS or not 0 <= shard < 8:
        raise ValueError('Vocabulary tile index out of bounds')
    return (group % 24)*8+shard, group//24


def weight_rectangles(rows_per_chunk=1):
    if not isinstance(rows_per_chunk, int) or isinstance(rows_per_chunk, bool) or not 1 <= rows_per_chunk <= 49:
        raise ValueError('Load chunk height must be in [1,49]')
    for y in range(0, 49, rows_per_chunk):
        yield (0, y, 192, min(rows_per_chunk, 49-y))
    yield (0, 49, 88, 1)


def identity_values():
    import numpy as np
    values = np.zeros((HEIGHT, WIDTH), dtype='<u4')
    for group in range(GROUPS):
        for shard in range(8):
            x, y = point(group, shard)
            values[y, x] = group
    return values


def make_layout(initialization='sdk', *, origin=(0, 0), fragment=False, device_control=False,
                code_file='pe.csl', first_layer=(62, 0), coordinate_identity=False, weight_storage='u32', static_data=None):
    # Keep the standalone emitter importable in its existing environments.
    sys.path.insert(0, str(ROOT / 'src'))
    from csl_llm.static_data import bindings
    data_bindings = bindings(static_data)
    remaining_bindings = set(data_bindings)
    if data_bindings and (initialization != 'sdk' or weight_storage != 'u8' or not device_control):
        raise ValueError('Explicit native data bindings require the model SDK/u8 mode')
    if weight_storage not in ('u32', 'u8') or (weight_storage == 'u8' and not device_control):
        raise ValueError('Native u8 storage is opt-in for the model vocabulary program')
    if weight_storage == 'u8' and initialization != 'sdk':
        raise ValueError('Native byte static initializers require a separately qualified generator')
    if coordinate_identity and (not device_control or not fragment or origin != (0, 160)):
        raise ValueError('Coordinate identity is restricted to the full-model vocabulary region')
    if initialization not in ('sdk', 'static'):
        raise ValueError('Unknown weight initialization mode')
    bx, by = origin
    if any(type(v) is not int or v < 0 for v in (*origin, *first_layer)):
        raise ValueError('Coordinates must be nonnegative integers')
    if not fragment and origin != (0, 0):
        raise ValueError('Translated vocabulary must be a layout fragment')
    if not code_file.endswith('.csl') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.' for c in code_file):
        raise ValueError('Code file must be a local CSL filename')
    lines = ['const memcpy=@import_module("<memcpy/get_params>",.{.width=193,.height=50});',
             'layout{@set_rectangle(193,50);'] if not fragment else []
    for y in range(HEIGHT):
        for x in range(WIDTH):
            group = y*24+x//8
            matrix = x < 192 and group < GROUPS
            controller = (x, y) == CONTROLLER
            weight = f'weights_{x}_{y}.csl' if matrix and initialization == 'static' else '<empty>'
            data = data_bindings.get((bx+x, by+y))
            if data is not None:
                if (not (matrix or controller) or
                        (data.weight_file is not None) != matrix or
                        (data.auxiliary_file is not None) != controller):
                    raise ValueError(f'Static data does not match vocabulary role at {(bx+x,by+y)}')
                weight = data.weight_file or '<empty>'
                remaining_bindings.remove((bx+x, by+y))
            params = (f'.memcpy_params=memcpy.get_params({bx+x}),.matrix={str(matrix).lower()},'
                      f'.controller={str(controller).lower()},'
                      f'.shard={x%8 if matrix else 0},.controller_x={bx+192},.controller_y={by},.weight_file="{weight}",'
                      f'.static_weights={str(initialization == "static" or (data is not None and matrix)).lower()}')
            if data is not None and controller:
                params += f',.static_norms=true,.auxiliary_file="{data.auxiliary_file}"'
            if device_control:
                params += f',.first_layer_x={first_layer[0]},.first_layer_y={first_layer[1]}'
            if coordinate_identity:
                params += ',.coordinate_identity=true'
            if weight_storage == 'u8':
                params += ',.weight_type=u8'
            lines.append(f'@set_tile_code({bx+x},{by+y},"{code_file}",.{{{params}}});')
    lines.append(f'@set_color_config({bx+192},{by},@get_color(3),.{{.routes=.{{.rx=.{{RAMP}},.tx=.{{WEST}}}}}});')
    if device_control:
        lines.append(f'@set_color_config({bx+192},{by},@get_color(4),.{{.routes=.{{.rx=.{{RAMP}},.tx=.{{WEST}}}}}});')
    for x in range(192):
        for y in range(HEIGHT):
            receiving = y*24+x//8 < GROUPS
            if not receiving:
                continue
            directions = (['WEST'] if y == 0 and x > 0 else [])
            if y+1 < HEIGHT and (y+1)*24+x//8 < GROUPS:
                directions.append('SOUTH')
            if receiving:
                directions.append('RAMP')
            # Stop at the last real group; padding tiles have no input route.
            routes = '.rx=.{' + ('EAST' if y == 0 else 'NORTH') + '},.tx=.{' + ','.join(directions) + '}'
            filtering = (f',.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter={(896-(x%8)*112)%896},.max_counter=111,.limit1=895}}'
                         if receiving else '')
            lines.append(f'@set_color_config({bx+x},{by+y},@get_color(3),.{{.routes=.{{{routes}}}{filtering}}});')
            if device_control:
                lines.append(f'@set_color_config({bx+x},{by+y},@get_color(4),.{{.routes=.{{{routes}}}}});')
    if remaining_bindings:
        raise ValueError('Static data lies outside this vocabulary region')
    if fragment:
        return '\n'.join(lines)+'\n'
    for name, kind in [('input', 'f32'), ('result', 'f32'), ('lookup', 'f32'), ('best', 'f32'),
                       ('winner', 'u32'), ('control', 'u32'), ('progress', 'u32'), ('timing', 'u16'), ('weights', weight_storage), ('identity', 'u32')]:
        lines.append(f'@export_name("{name}",[*]{kind},false);')
    if device_control:
        for name in ['norms', 'summary', 'prompt', 'generated']:
            lines.append(f'@export_name("{name}",[*]u32,false);')
        lines.append('@export_name("reset_model",fn()void);')
    for name in ['initialize', 'prepare', 'compute']:
        lines.append(f'@export_name("{name}",fn()void);')
    return '\n'.join(lines+['}'])+'\n'


def execution_options(channels=1, threads=8):
    if not isinstance(channels, int) or isinstance(channels, bool) or not 1 <= channels <= 16:
        raise ValueError('Pinned SDK permits1..16 memcpy channels for this50-row layout')
    if not isinstance(threads, int) or isinstance(threads, bool) or not 1 <= threads <= 18:
        raise ValueError('Simulator threads must fit the18 physical cores on workstation')
    # Pinned cslc wrapper lines142..158; uneven first band absorbs the remainder.
    ceil_height = (HEIGHT+channels-1)//channels
    normal = HEIGHT//channels if ceil_height*(channels-1) >= HEIGHT else ceil_height
    heights = [HEIGHT-normal*(channels-1)]+[normal]*(channels-1)
    cursor, bands = 0, []
    for height in heights:
        bands.append([cursor, height]);cursor += height
    assert cursor == HEIGHT and all(height > 0 for _, height in bands)
    return dict(memcpy_channels=channels, simulator_threads=threads, channel_row_bands=bands)


def prepare(source_root, parallelism, initialization='sdk', rows_per_chunk=1, channels=1, threads=8):
    if parallelism < 1:
        raise ValueError('Compiler parallelism must be positive')
    if initialization not in ('sdk', 'static'):
        raise ValueError('Unknown initialization mode')
    rectangles = list(weight_rectangles(rows_per_chunk))
    options = execution_options(channels, threads)
    import numpy as np
    import torch
    from safetensors.torch import load_file
    sys.path.insert(0, str(source_root / 'tools'))
    from bf16_transport import pack_bf16_pairs
    torch.set_num_threads(4)
    model = ROOT / 'models/qwen2.5-0.5b-7ae5576/model.safetensors'
    trace = ROOT / 'evidence/reference-f32-greedy-20260908T0554/arithmetic/step-000.npz'
    state = load_file(model)
    matrix = state['model.embed_tokens.weight'].float().numpy()
    assert matrix.shape == (151936, 896)
    reference = np.load(trace)
    canonical = reference['model.norm'].reshape(-1, 896)[-1].copy()
    official = reference['logits'].reshape(-1, 151936)[-1].copy()
    # Independent original-weight full contraction before any candidate output.
    expected = (torch.from_numpy(matrix) @ torch.from_numpy(canonical)).numpy()
    delta = expected.astype(np.float64)-official.astype(np.float64)
    check = dict(max_abs=float(np.max(np.abs(delta))),
                 relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(official.astype(np.float64))),
                 relative_peak=float(np.max(np.abs(delta))/np.max(np.abs(official))))
    assert check['max_abs'] <= .01 and check['relative_l2'] <= 2e-4 and check['relative_peak'] <= 5e-4
    out = ROOT / 'evidence' / ('resident-vocabulary-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    np.save(out / 'identity.npy', identity_values())
    packed_weights = (np.memmap(out / 'weights.u32.bin', mode='w+', dtype='<u4', shape=(GROUPS*8, 7168))
                      if initialization == 'sdk' else None)
    for group in range(GROUPS):
        for shard in range(8):
            x, y = point(group, shard)
            tile = matrix[group*128:(group+1)*128, shard*112:(shard+1)*112].T.copy()
            words = pack_bf16_pairs((tile.view(np.uint32)>>16).astype(np.uint16))
            if packed_weights is not None:
                packed_weights[group*8+shard] = words
            else:
                (out / f'weights_{x}_{y}.csl').write_text('const values=[7168]u32{'+','.join(f'0x{int(w):08x}' for w in words)+'};\n')
    if packed_weights is not None:
        packed_weights.flush()
        del packed_weights
    token_ids = [0, 151645, 151935]
    np.savez(out / 'inputs.npz', inputs=np.stack([canonical, np.zeros(896, np.float32), canonical]),
             expected=np.stack([expected, np.zeros(151936, np.float32), expected]),
             official=np.stack([official, np.zeros(151936, np.float32), official]),
             lookup=matrix[token_ids], token_ids=np.array(token_ids, np.uint32))
    for source, target in [(Path(__file__), 'driver.py'), (source_root / 'tools/sdk_probe.py', 'executor.py'),
                           (source_root / 'csl/programs/resident_vocabulary/pe.csl', 'pe.csl'),
                           (source_root / 'csl/kernels/local_gemv_bf16_f32_colmajor.csl', 'kernel.csl'),
                           (source_root / 'csl/kernels/tied_embedding_f32.csl', 'embedding.csl'),
                           (source_root / 'csl/kernels/argmax_f32.csl', 'argmax.csl'),
                           (source_root / 'csl/runtime/line_allreduce_f32.csl', 'line.csl'),
                           (source_root / 'csl/runtime/filtered_input.csl', 'input.csl'),
                           (source_root / 'csl/runtime/packets.csl', 'packets.csl'),
                           (source_root / 'configs/precision.json', 'precision.json')]:
        shutil.copy2(source, out / target)
    (out / 'layout.csl').write_text(make_layout(initialization))
    config = dict(compile_parallelism=parallelism, model_sha256=digest(model), trace_sha256=digest(trace),
                  reference_check=check, matrix_tiles=GROUPS*8, vocabulary=151936,
                  initialization=initialization, weight_rectangles=rectangles,
                  group_identity='Once-loaded u32 metadata, not a compiler parameter',
                  scope='Complete original151936x896 tied embedding/LM head, three exact full896lookups and three full-vocabulary projections/deviceargmax. Already-normalized reference inputs; no finalRMS/24layers/autoregressive generation claim. Standalone controller at192,0, distinct from future whole-model integration.')
    config.update(options)
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=SDK_SHA, files={p.name: digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def rebuild_fixture(before, source_root, parallelism, rows_per_chunk, channels=1, threads=8):
    """Reuse verified oracle/payload bytes while freezing a new program variant."""
    import numpy as np
    if parallelism < 1:
        raise ValueError('Compiler parallelism must be positive')
    rectangles = list(weight_rectangles(rows_per_chunk))
    options = execution_options(channels, threads)
    manifest = json.loads((before / 'manifest.json').read_text())
    config = json.loads((before / 'config.json').read_text())
    if config.get('initialization') != 'sdk' or config.get('matrix_tiles') != 9496:
        raise ValueError('Parent must be a complete SDK-initialized vocabulary fixture')
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(before / name) != sha:
            raise ValueError(f'Invalid parent input {name}')
    out = ROOT / 'evidence' / ('resident-vocabulary-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    for name in manifest['files']:
        shutil.copy2(before / name, out / name)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source_root / 'csl/programs/resident_vocabulary/pe.csl', out / 'pe.csl')
    (out / 'layout.csl').write_text(make_layout('sdk'))
    np.save(out / 'identity.npy', identity_values())
    config.update(compile_parallelism=parallelism, weight_rectangles=rectangles,
                  group_identity='Once-loaded u32 metadata, not a compiler parameter',
                  parent_manifest_sha256=digest(before / 'manifest.json'))
    config.update(options)
    (out / 'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
                                                       files={p.name:digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform
    verify(root)
    os.chdir(root)
    config = json.loads((root / 'config.json').read_text())
    parallelism = config['compile_parallelism']
    options = execution_options(config.get('memcpy_channels', 1), config.get('simulator_threads', 8))
    if 'channel_row_bands' in config:
        assert config['channel_row_bands'] == options['channel_row_bands']
    assert isinstance(parallelism, int) and not isinstance(parallelism, bool) and parallelism > 0
    start = time.monotonic()
    def stage(name):
        row = dict(stage=name, elapsed_seconds=time.monotonic()-start)
        (root / 'stage.json').write_text(json.dumps(row)+'\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        print(name, flush=True)
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=200,52', '--fabric-offsets=4,1',
               '-o=out', '--memcpy', f"--channels={options['memcpy_channels']}", '--dump-dsr-alloc-graph', f'--max-parallelism={parallelism}']
    (root / 'compile-command.json').write_text(json.dumps(command)+'\n')
    stage('compile')
    subprocess.run(command, check=True)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True, num_threads=options['simulator_threads'], dump_core=True), SdkTarget.WSE3))
    ids = {name: runner.get_id(name) for name in ['input', 'result', 'lookup', 'best', 'winner', 'control', 'progress', 'timing', 'weights', 'identity']}
    def transfer(name, x, y, width, height, count, values=None):
        read = values is None
        operation = 'd2h' if read else 'h2d'
        stage(f'{operation}-{name}-{x}-{y}-{width}-{height}-{count}-begin')
        if read:
            values = np.zeros(width*height*count, np.uint32 if name in ('winner', 'control', 'progress', 'timing', 'weights', 'identity') else np.float32)
        kwargs = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT if name == 'timing' else MemcpyDataType.MEMCPY_32BIT,
                      order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        if read:
            runner.memcpy_d2h(values, ids[name], x, y, width, height, count, **kwargs)
        else:
            runner.memcpy_h2d(ids[name], values, x, y, width, height, count, **kwargs)
        stage(f'{operation}-{name}-end')
        return values
    reports, epoch = [], 0
    data = np.load(root / 'inputs.npz')
    stage('load');runner.load()
    stage('run');runner.run()
    try:
        runner.launch('initialize', nonblock=False)
        identities = np.load(root / 'identity.npy')
        assert identities.shape == (HEIGHT, WIDTH) and identities.dtype == np.dtype('<u4')
        transfer('identity', 0, 0, WIDTH, HEIGHT, 1, identities.reshape(-1))
        actual_identity = transfer('identity', 0, 0, WIDTH, HEIGHT, 1)
        np.save(root / 'actual-identity.npy', actual_identity)
        np.testing.assert_array_equal(actual_identity, identities.reshape(-1))
        (root / 'identity-load.json').write_text(json.dumps(dict(exact=True, rectangle=[0, 0, WIDTH, HEIGHT],
                                                                 source_sha256=digest(root / 'identity.npy'),
                                                                 actual_sha256=digest(root / 'actual-identity.npy')), indent=2)+'\n')
        if config['initialization'] == 'sdk':
            # Keep the complete backing map alive until runner.stop(). No
            # temporary H2D arrays are reused while SDK work may be pending.
            assert (root / 'weights.u32.bin').stat().st_size == GROUPS*8*7168*4
            packed_weights = np.memmap(root / 'weights.u32.bin', mode='c', dtype='<u4', shape=(GROUPS*8, 7168))
            loaded_tiles = 0
            for block, (x, y, width, height) in enumerate(config['weight_rectangles']):
                assert x == 0 and y*192 == loaded_tiles
                assert (y < 49 and width == 192 and 1 <= height <= 49-y) or (y == 49 and width == 88 and height == 1)
                count = width*height
                values = packed_weights[loaded_tiles:loaded_tiles+count].reshape(-1)
                stage(f'weight-block-{block}-load')
                transfer('weights', x, y, width, height, 7168, values)
                stage(f'weight-block-{block}-readback')
                actual = transfer('weights', x, y, width, height, 7168)
                path = root / f'loaded-weights-{block}.npy'
                np.save(path, actual)
                np.testing.assert_array_equal(actual, values)
                with (root / 'weight-loads.jsonl').open('a') as stream:
                    stream.write(json.dumps(dict(block=block, rectangle=[x, y, width, height],
                                                 first_tile=loaded_tiles, tiles=count, words_per_tile=7168,
                                                 source_bytes_sha256=hashlib.sha256(values.tobytes()).hexdigest(),
                                                 actual_file=path.name, actual_sha256=digest(path), exact=True))+'\n')
                loaded_tiles += count
            assert loaded_tiles == GROUPS*8
            stage('all-original-weights-exact-readback')
        for case in range(3):
            for mode in range(2):
                epoch += 1
                stage(f'case-{case}-mode-{mode}')
                metadata = np.tile(np.array([mode, data['token_ids'][case]], np.uint32), WIDTH*HEIGHT)
                transfer('control', 0, 0, WIDTH, HEIGHT, 2, metadata)
                if mode:
                    transfer('input', *CONTROLLER, 1, 1, 896, data['inputs'][case].copy())
                runner.launch('prepare', nonblock=False)
                ready = transfer('progress', 0, 0, WIDTH, HEIGHT, 1)
                np.testing.assert_array_equal(ready, np.full(WIDTH*HEIGHT, epoch*2-1, np.uint32))
                runner.launch('compute', nonblock=False)
                actual = {'ready': ready}
                if mode == 0:
                    actual['lookup'] = transfer('lookup', *CONTROLLER, 1, 1, 896)
                    np.save(root / f'lookup-{case}.npy', actual['lookup'])
                else:
                    actual['best'] = transfer('best', *CONTROLLER, 1, 1, 1)
                    np.save(root / f'best-{case}.npy', actual['best'])
                    actual['winner'] = transfer('winner', *CONTROLLER, 1, 1, 1)
                    np.save(root / f'winner-{case}.npy', actual['winner'])
                    columns = []
                    for x in range(0, 192, 8):
                        height = (GROUPS-1-x//8)//24+1
                        column = np.full((HEIGHT, 128), np.nan, np.float32)
                        raw = transfer('result', x, 0, 1, height, 128).reshape(height, 128)
                        column_path = root / f'logits-{case}-column-{x}.npy'
                        np.save(column_path, raw)
                        with (root / 'column-outputs.jsonl').open('a') as stream:
                            stream.write(json.dumps(dict(case=case, x=x, height=height,
                                                         file=column_path.name, sha256=digest(column_path)))+'\n')
                        column[:height] = raw
                        columns.append(column)
                    actual['logits'] = np.stack(columns, axis=1).reshape(-1)[:151936]
                actual['progress'] = transfer('progress', 0, 0, WIDTH, HEIGHT, 1)
                actual['timing'] = transfer('timing', *CONTROLLER, 1, 1, 6)
                path = root / f'actual-{case}-{mode}.npz'
                np.savez(path, **actual)
                np.testing.assert_array_equal(actual['progress'], np.full(WIDTH*HEIGHT, epoch*2, np.uint32))
                checks = {}
                if mode == 0:
                    np.testing.assert_array_equal(actual['lookup'].view(np.uint32), data['lookup'][case].view(np.uint32))
                    checks['lookup_bits_exact'] = True
                else:
                    assert np.isfinite(actual['logits']).all()
                    for name in ['expected', 'official']:
                        expected = data[name][case].astype(np.float64)
                        delta = actual['logits'].astype(np.float64)-expected
                        check = dict(max_abs=float(np.max(np.abs(delta))), relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(expected), 1e-30)),
                                     relative_peak=float(np.max(np.abs(delta))/max(np.max(np.abs(expected)), 1e-30)))
                        assert check['max_abs'] <= .01 and check['relative_l2'] <= 2e-4 and check['relative_peak'] <= 5e-4, check
                        assert int(actual['winner'][0]) == int(np.argmax(expected))
                        checks[name] = check
                    index = int(actual['winner'][0])
                    assert actual['best'][0] == actual['logits'][index]
                    assert index == int(np.argmax(actual['logits']))
                    if case == 1:
                        np.testing.assert_array_equal(actual['logits'], np.zeros(151936, np.float32))
                        assert index == 0
                    if case == 2:
                        first = np.load(root / 'actual-0-1.npz')
                        np.testing.assert_array_equal(actual['logits'].view(np.uint32), first['logits'].view(np.uint32))
                t = actual['timing'].astype(np.uint64)
                assert np.all(t < 65536)
                ticks = [sum(int(t[k+i]) << (16*i) for i in range(3)) for k in [0, 3]]
                cycles = (ticks[1]-ticks[0]) & ((1 << 48)-1)
                assert 0 < cycles < 1 << 40
                reports.append(dict(case=case, mode=mode, checks=checks, controller_cycles=cycles, actual_sha256=digest(path)))
                (root / 'results.json').write_text(json.dumps(dict(success=False, cases=reports), indent=2)+'\n')
    finally:
        stage('stop');runner.stop()
    (root / 'results.json').write_text(json.dumps(dict(success=True, cases=reports, scope=config['scope']), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    action.add_argument('--from-fixture', type=Path)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--compile-parallelism', type=int, default=8)
    parser.add_argument('--initialization', choices=['sdk', 'static'], default='sdk')
    parser.add_argument('--load-rows-per-chunk', type=int, default=1)
    parser.add_argument('--memcpy-channels', type=int, default=1)
    parser.add_argument('--simulator-threads', type=int, default=8)
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    if args.project_root:
        ROOT = args.project_root.resolve()
    if args.from_fixture:
        rebuild_fixture(args.from_fixture.resolve(), args.source_root.resolve() if args.source_root else ROOT,
                        args.compile_parallelism, args.load_rows_per_chunk, args.memcpy_channels, args.simulator_threads)
    elif args.prepare:
        prepare(args.source_root.resolve() if args.source_root else ROOT, args.compile_parallelism, args.initialization,
                args.load_rows_per_chunk, args.memcpy_channels, args.simulator_threads)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), args.timeout)

"""Pinned SDK streaming D2H: ordered tagged columns, completion and reset.

First qualify the small fabric. Full193x50 topology is a separate bounded run.
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


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(source, project, full=False):
    width, height, tail = (193, 50, 11) if full else (17, 2, 2)
    sequences = [[0, 184, 0], [184]] if full else [[0, 8, 16, 0], [16, 0]]
    out = project / 'evidence' / ('stream-output-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    layout = ['param MEMCPYD2H_DATA_1_ID:i16;',
              f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height},.MEMCPYD2H_1=MEMCPYD2H_DATA_1_ID}});',
              f'layout{{@set_rectangle({width},{height});']
    for y in range(height):
        for x in range(width):
            active = x%8 == 0 and x < (192 if full else 17) and (y < height-1 or x//8 < tail)
            layout.append(f'@set_tile_code({x},{y},"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),.root={str(active).lower()},.width={width}}});')
    layout += ['@export_name("progress",[*]u32,false);', '@export_name("prepare",fn(i16)void);',
               '@export_name("export_output",fn(i16)void);', '}']
    (out / 'layout.csl').write_text('\n'.join(layout)+'\n')
    for path, name in [(Path(__file__), 'driver.py'), (source / 'tools/sdk_probe.py', 'executor.py'),
                       (source / 'csl/programs/stream_output/pe.csl', 'pe.csl')]:
        shutil.copy2(path, out / name)
    examples = project.parent / 'sdk-examples/tutorials/gemv-09-streaming'
    provenance = {name: digest(examples / name) for name in ['layout.csl', 'pe_program.csl', 'run.py', 'commands_wse3.sh']}
    (out / 'config.json').write_text(json.dumps(dict(width=width, height=height,
        last_row_groups=tail, sequences=sequences, output_color=8, output_queue=5,
        completion_task=14, simulator_threads=8 if full else 1,
        timeout_seconds=900 if full else 120, rss_limit_kib=(8 if full else 1)*1024*1024,
        sdk_example=str(examples), example_files=provenance,
        scope='Tagged streaming D2H transport only. Scalar selected-column export, one source per row, '
              'nonblocking launch then exact128word per-source stream receive/task_wait and source '
              'completion counters; full application progress before reset/exit. No neural kernel, '
              'hardware bandwidth or full vocabulary numerical acceptance.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
        files={p.name: digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    from executor import verify
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    verify(root)
    os.chdir(root)
    config = json.loads((root / 'config.json').read_text())
    width, height = config['width'], config['height']
    start = time.monotonic()

    def stage(name):
        value = dict(stage=name, elapsed_seconds=time.monotonic()-start)
        (root / 'stage.json').write_text(json.dumps(value)+'\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        print(name, flush=True)

    command = ['cslc', 'layout.csl', '--arch=wse3', f'--fabric-dims={width+7},{height+2}',
               '--fabric-offsets=4,1', '--params=MEMCPYD2H_DATA_1_ID:8', '-o=out',
               '--memcpy', '--channels=1', '--max-parallelism=1', '--dump-dsr-alloc-graph']
    (root / 'compile-command.json').write_text(json.dumps(command)+'\n')
    stage('compile')
    subprocess.run(command, check=True)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True,
        num_threads=config['simulator_threads'], dump_core=True), SdkTarget.WSE3))
    progress_id = runner.get_id('progress')
    records = []

    def read_progress(x, rows, columns=1):
        result = np.zeros(rows*columns, np.uint32)
        runner.memcpy_d2h(result, progress_id, x, 0, columns, rows, 1, streaming=False,
            order=MemcpyOrder.ROW_MAJOR, data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
        return result

    try:
        stage('load');runner.load()
        stage('run');runner.run()
        for tag, columns in enumerate(config['sequences'], 1):
            stage(f'prepare-{tag}')
            runner.launch('prepare', np.int16(tag), nonblock=False)
            for call, column in enumerate(columns, 1):
                rows = height if column//8 < config['last_row_groups'] else height-1
                actual = np.zeros(rows*128, np.uint32)
                stage(f'export-{tag}-{call}-{column}')
                begin = time.monotonic()
                task = runner.launch('export_output', np.int16(column), nonblock=True)
                runner.memcpy_d2h(actual, config['output_color'], column, 0, 1, rows, 128,
                    streaming=True, order=MemcpyOrder.ROW_MAJOR,
                    data_type=MemcpyDataType.MEMCPY_32BIT, nonblock=False)
                runner.task_wait(task)
                elapsed = time.monotonic()-begin
                ready = read_progress(column, rows)
                file = root / f'actual-{tag}-{call}.npz'
                np.savez(file, output=actual.reshape(rows, 128), progress=ready)
                expected = ((np.uint32(tag)<<24) | (np.arange(rows, dtype=np.uint32)[:, None]<<16)
                            | np.uint32(column<<8) | np.arange(128, dtype=np.uint32)[None, :])
                np.testing.assert_array_equal(actual.reshape(rows, 128), expected)
                np.testing.assert_array_equal(ready, np.full(rows, call, np.uint32))
                records.append(dict(tag=tag, call=call, column=column, rows=rows,
                    words=actual.size, api_elapsed_seconds=elapsed, actual_sha256=digest(file)))
            stage(f'global-completion-{tag}')
            global_progress = read_progress(0, height, width)
            np.save(root / f'global-progress-{tag}.npy', global_progress)
            np.testing.assert_array_equal(global_progress, np.full(width*height, len(columns), np.uint32))
    finally:
        stage('stop');runner.stop()
    (root / 'results.json').write_text(json.dumps(dict(success=True, cases=records,
        scope=config['scope'], timing_scope='Host API launch/receive/task_wait interval; not device kernel cycles or hardware bandwidth.'), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--full-fabric', action='store_true')
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.source_root.resolve(), args.project_root.resolve(), args.full_fabric)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        config = json.loads((args.execute / 'config.json').read_text())
        execute(args.execute.resolve(), config['timeout_seconds'], rss_limit_kib=config['rss_limit_kib'])

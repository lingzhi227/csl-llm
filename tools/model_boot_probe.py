"""Full-layout SDK initialization and global prepare; no neural computation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def prepare(audited, source, project):
    sys.path.insert(0, str(source / 'tools'))
    from sdk_probe import verify, sha
    manifest = verify(audited)
    execution = json.loads((audited / 'execution.json').read_text())
    result = json.loads((audited / 'results.json').read_text())
    assert execution['success'] and result['success'] and result['offline_only']
    assert execution['manifest_sha256'] == sha(audited / 'manifest.json')
    assert execution['results_sha256'] == sha(audited / 'results.json')
    for name, expected in result['report_sha256'].items():
        assert sha(audited / name) == expected
        assert json.loads((audited / name).read_text())['passed']
    out = project / 'evidence' / ('model-boot-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    shutil.copytree(audited / 'compiled', out / 'compiled')
    for directory in ('tools', 'src'):
        shutil.copytree(audited / directory, out / directory,
                        ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(source / 'tools/sdk_core.py', out / 'tools/sdk_core.py')
    shutil.copy2(source / 'src/csl_llm/coordinate_identity.py', out / 'src/csl_llm/coordinate_identity.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    shutil.copy2(Path(__file__), out / 'driver.py')
    for name in ('manifest.json', 'execution.json', 'results.json'):
        shutil.copy2(audited / name, out / ('audit-'+name))
    for name in result['report_sha256']:
        shutil.copy2(audited / name, out / name)
    (out / 'config.json').write_text(json.dumps(dict(
        audited_bundle=audited.name, audit_execution_sha256=sha(audited / 'execution.json'),
        simulator_threads=16, timeout_seconds=1800, rss_limit_kib=24*1024*1024,
        prompt_token=22, generation_limit=0, expected_progress=1,
        scope='One fresh full193x210 SDK load/run/initialize/reset/prepare instance. '
              'No compute invocation or neural arithmetic. All40530actual progress markers must be1; '
              'matrix/KV/guard identities, initial weights/norms/bias/frequencies and decoder logical '
              'status checked from stopped core. Normal controller D2H is bound to core values. '
              'Only12matrices/25normcontrollers/3biasroots have original data; other matrices/biases '
              'are zero. Not full original deployment, physical KV capacity, generation or hardware performance.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path.insert(0, str(root / 'tools'))
    sys.path.insert(0, str(root / 'src'))
    import numpy as np
    from executor import verify, sha
    from sdk_core import StoppedCore
    from model_initializer_audit import expected_symbols
    from csl_llm.decoder_layout import tiles
    from csl_llm.coordinate_identity import decoder_identity, vocabulary_identity
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    manifest = verify(root)
    config = json.loads((root / 'config.json').read_text())
    execution = json.loads((root / 'audit-execution.json').read_text())
    result = json.loads((root / 'audit-results.json').read_text())
    source_manifest = json.loads((root / 'audit-manifest.json').read_text())
    assert sha(root / 'audit-execution.json') == config['audit_execution_sha256']
    assert execution['success'] and execution['sdk_sha256'] == manifest['sdk_sha256']
    assert execution['manifest_sha256'] == sha(root / 'audit-manifest.json')
    assert execution['results_sha256'] == sha(root / 'audit-results.json')
    for name, expected in result['report_sha256'].items():
        assert sha(root / name) == expected
    for name, expected in source_manifest['files'].items():
        if name.startswith('compiled/'):
            assert sha(root / name) == expected
    compiled = root / 'compiled'
    assert result['source_execution_sha256'] == sha(compiled / 'execution.json')
    initial = json.loads((root / 'initializers.json').read_text())
    assert initial['passed'] and initial['application_coordinates'] == 40530
    compiled_files = {name.removeprefix('compiled/out/'): value
        for name, value in manifest['files'].items() if name.startswith('compiled/out/')}
    assert compiled_files
    directory = root / 'runtime'
    directory.mkdir()
    shutil.copytree(compiled / 'out', directory / 'out')
    (root / 'compiled-files.json').write_text(json.dumps(compiled_files, indent=2)+'\n')
    started = time.monotonic()

    def stage(name):
        value = dict(stage=name, elapsed_seconds=time.monotonic()-started,
                     utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        (root / 'stage.json').write_text(json.dumps(value)+'\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(value)+'\n')
        print(name, flush=True)

    os.chdir(directory)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(
        suppress_trace=True, num_threads=config['simulator_threads'], dump_core=True), SdkTarget.WSE3))
    ids = {name: runner.get_id(name) for name in ('progress', 'control', 'summary', 'prompt')}
    normal = {}

    def transfer(name, count, value=None):
        stage(('h2d-' if value is not None else 'd2h-')+name)
        options = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                       order=MemcpyOrder.ROW_MAJOR, nonblock=False)
        if value is not None:
            runner.memcpy_h2d(ids[name], np.asarray(value, np.uint32).ravel(),
                              192, 160, 1, 1, count, **options)
        else:
            value = np.zeros(count, np.uint32)
            runner.memcpy_d2h(value, ids[name], 192, 160, 1, 1, count, **options)
            normal[name] = value
            # Preserve each completed D2H even if a later check fails.
            np.savez(directory / 'normal-d2h.npz', **normal)
            return value

    try:
        stage('load'); runner.load()
        stage('run'); runner.run()
        stage('initialize'); runner.launch('initialize', nonblock=False)
        transfer('prompt', 1, [config['prompt_token']])
        transfer('control', 2, [1, config['generation_limit']])
        stage('reset-model'); runner.launch('reset_model', nonblock=False)
        stage('prepare'); runner.launch('prepare', nonblock=False)
        np.testing.assert_array_equal(transfer('progress', 1), [1])
        np.testing.assert_array_equal(transfer('control', 2), [1, 0])
        np.testing.assert_array_equal(transfer('summary', 7), np.zeros(7, np.uint32))
        np.testing.assert_array_equal(transfer('prompt', 1), [config['prompt_token']])
    finally:
        stage('stop'); runner.stop()

    stage('verify-runtime-artifacts')
    for name, expected in compiled_files.items():
        assert sha(directory / 'out' / name) == expected
    artifacts = {str(path.relative_to(directory)): sha(path)
        for path in (directory / 'out').rglob('*') if path.is_file()}
    (directory / 'runtime-artifacts.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    stage('actual-core-read')
    core = StoppedCore(directory / 'out.core', directory / 'out/bin')
    expectations = dict(expected_symbols(compiled))
    owners = {(tile.x, tile.y): tile for layer in range(24) for tile in tiles(layer)}
    progress = np.empty((210, 193), np.uint32)
    decoder_ids = np.empty((160, 193, 3), np.uint32)
    vocabulary_ids = np.empty((50, 193), np.uint32)
    status = np.empty((160, 193, 2), np.uint32)

    def read(x, y, name, count):
        return np.frombuffer(core.read(x+4, y+1, name, count*4), dtype='<u4').copy()

    for (x, y), expected in expectations.items():
        for name, raw in expected.items():
            assert core.read(x+4, y+1, name, len(raw)) == raw, (x, y, name)
        progress[y, x] = read(x, y, 'progress', 1)[0]
        assert progress[y, x] == 1, (x, y, 'global prepare not completed')
        if y < 160:
            tile = owners.get((x, y))
            wanted = decoder_identity(tile.operation, x, y) if tile else (0, 0, 0)
            decoder_ids[y, x] = read(x, y, 'identity', 3)
            np.testing.assert_array_equal(decoder_ids[y, x], wanted)
            status[y, x] = read(x, y, 'status', 2)
            np.testing.assert_array_equal(status[y, x], [0, 0])
        else:
            matrix = x < 192 and (y-160)*192+x < 9496
            vocabulary_ids[y-160, x] = read(x, y, 'identity', 1)[0]
            assert vocabulary_ids[y-160, x] == (vocabulary_identity(x, y) if matrix else 0)
            np.testing.assert_array_equal(read(x, y, 'control', 2), [1, 0] if (x, y) == (192, 160) else [0, 0])
    for name, count in [('progress', 1), ('control', 2), ('summary', 7)]:
        np.testing.assert_array_equal(read(192, 160, name, count), normal[name])
    # Prompt D2H is checked as host-loaded metadata. It is not a neural result,
    # and no unobserved private prompt symbol name is inferred for the core.
    np.savez(directory / 'actual.npz', progress=progress, decoder_identity=decoder_ids,
             vocabulary_identity=vocabulary_ids, decoder_status=status)
    verify(root)
    (root / 'results.json').write_text(json.dumps(dict(success=True, neural_execution=False,
        application_coordinates=40530, matrix_tiles=34552, kv_tiles=3168,
        all_initializers_exact=True, all_coordinate_identities_exact=True,
        all_global_prepare_markers_one=True, all_decoder_logical_status_zero=True,
        normal_d2h_core_progress_control_summary_exact=True,
        audit_execution_sha256=config['audit_execution_sha256'],
        compiled_files_sha256=sha(root / 'compiled-files.json'),
        runtime_artifacts_sha256=sha(directory / 'runtime-artifacts.json'),
        actual_sha256=sha(directory / 'actual.npz'),
        normal_d2h_sha256=sha(directory / 'normal-d2h.npz'),
        core_sha256=sha(directory / 'out.core'), core_bytes=(directory / 'out.core').stat().st_size,
        scope=config['scope']), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-audit', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT / 'support/model_boot')
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.from_audit:
        prepare(args.from_audit.resolve(), args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 1800, rss_limit_kib=24*1024*1024)

"""Freeze and execute the original position-zero two-layer device chain.

The host supplies only the original embedding input. No host layer relay or
neural computation occurs. Numerical oracles are frozen independent fixtures.
Actual stopped-core reading is diagnostic, separate from production host I/O.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = '9c2eb93b26f396e71f8ca43b5501d0ea8408f8c2595f5818456332b1a1f19ab5'
REFERENCE_INPUTS = 'da7667460d25f5b70acd47ad0e1728b3ebbd79d9aabe86f3ab983b7a20b69a73'


def assembled_receipt(root):
    from sdk_probe import verify, sha
    manifest = verify(root)
    execution = json.loads((root / 'execution.json').read_text())
    result = json.loads((root / 'results.json').read_text())
    assert execution['success'] and result['success'] and result['passed'] and result['offline_only']
    assert execution['manifest_sha256'] == sha(root / 'manifest.json')
    assert execution['results_sha256'] == sha(root / 'results.json')
    assert execution['sdk_sha256'] == manifest['sdk_sha256']
    assert result['initializers']['passed'] and result['initializers']['original_matrix_tiles'] == 2088
    composition = result['composition']
    assert composition['passed'] and composition['coordinate_count'] == 2580
    for name, expected in composition['artifact_sha256'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert sha(root / 'out' / relative) == expected
    assert sha(root / 'symbols.json') == result['symbols_sha256']
    return manifest, result


def prepare(assembled, reference, source, project):
    sys.path[:0] = [str(source / 'tools'), str(source / 'src')]
    from sdk_probe import verify, sha
    from sdk_core import contract_symbols
    manifest, result = assembled_receipt(assembled)
    contract_symbols(json.loads((assembled / 'symbols.json').read_text()), result['composition'])
    verify(reference)
    assert sha(reference / 'manifest.json') == REFERENCE_MANIFEST
    assert sha(reference / 'inputs.npz') == REFERENCE_INPUTS
    reference_config = json.loads((reference / 'config.json').read_text())
    assert (reference_config['token_id'], reference_config['position'], reference_config['layers']) == (151644, 0, [0, 1])
    out = project / 'evidence' / ('decoder-chain-runtime-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    shutil.copytree(assembled, out / 'assembled', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(reference, out / 'reference', ignore=shutil.ignore_patterns('__pycache__'))
    names = ['tools/decoder_chain_validate.py', 'tools/decoder_chain_symbols.py',
             'tools/decoder_chain_assemble.py', 'tools/decoder_chain_audit_probe.py',
             'tools/model_initializer_audit.py', 'tools/model_symbols.py',
             'tools/sdk_core.py', 'tools/sdk_probe.py', 'src/csl_llm/__init__.py',
             'src/csl_llm/decoder_layout.py', 'src/csl_llm/regions.py']
    for name in names:
        destination = out / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, destination)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    shutil.copy2(source / 'configs/precision.json', out / 'precision.json')
    assert sha(out / 'precision.json') == sha(reference / 'precision.json')
    (out / 'config.json').write_text(json.dumps(dict(assembly_execution_sha256=sha(assembled / 'execution.json'),
        reference_manifest_sha256=REFERENCE_MANIFEST, reference_inputs_sha256=REFERENCE_INPUTS,
        timeout_seconds=1800, rss_limit_kib=8*1024*1024, simulator_threads=8,
        token_id=151644, position=0, layers=[0, 1],
        scope='Original two-layer device stream at causal position zero. '
              'Host only supplies initial embedding and reads final output/diagnostics. '
              'No full-model logits, cached second token, reset isolation, or generation acceptance.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    sys.path[:0] = [str(root / 'tools'), str(root / 'src')]
    from executor import verify, sha
    from sdk_core import StoppedCore
    from decoder_chain_assemble import complete_expected
    from decoder_chain_validate import validate
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    assert config['timeout_seconds'] == 1800 and config['rss_limit_kib'] == 8*1024*1024
    assert config['simulator_threads'] == 8 and config['position'] == 0
    assert sha(root / 'assembled/execution.json') == config['assembly_execution_sha256']
    _, assembly = assembled_receipt(root / 'assembled')
    assert sha(root / 'reference/manifest.json') == config['reference_manifest_sha256'] == REFERENCE_MANIFEST
    assert sha(root / 'reference/inputs.npz') == config['reference_inputs_sha256'] == REFERENCE_INPUTS
    reference = np.load(root / 'reference/inputs.npz')
    policy = json.loads((root / 'precision.json').read_text())['stage_gate']['absolute_or_normwise']
    assert policy == dict(max_abs=1e-5, both=dict(relative_l2=2e-4, relative_peak=5e-4))
    original = np.ascontiguousarray(reference['input'].reshape(-1))
    assert original.dtype == np.float32 and original.shape == (896,) and np.isfinite(original).all()
    directory = root / 'runtime'
    directory.mkdir()
    shutil.copytree(root / 'assembled/out', directory / 'out')
    compiled_files = assembly['composition']['artifact_sha256']
    (root / 'compiled-files.json').write_text(json.dumps(compiled_files, indent=2)+'\n')
    def verify_images():
        for name, expected in compiled_files.items():
            assert sha(directory / 'out' / name) == expected
    verify_images()
    start = time.monotonic()
    def stage(name):
        event = dict(stage=name, elapsed_seconds=time.monotonic()-start)
        (root / 'stage.json').write_text(json.dumps(event)+'\n')
        with (root / 'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(name, flush=True)
    os.chdir(directory)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True,
        num_threads=8, dump_core=True), SdkTarget.WSE3))
    ids = {name: runner.get_id(name) for name in ('input', 'output', 'progress')}
    options = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                   order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    def read(name, x, y, width, height, count):
        stage('d2h-'+name)
        values = np.zeros(width*height*count, np.float32 if name == 'output' else np.uint32)
        runner.memcpy_d2h(values, ids[name], x, y, width, height, count, **options)
        return values
    normal = {}
    try:
        stage('load'); runner.load()
        stage('run'); runner.run()
        stage('initialize'); runner.launch('initialize', nonblock=False)
        stage('reset-model'); runner.launch('reset_model', nonblock=False)
        stage('initial-embedding-h2d')
        runner.memcpy_h2d(ids['input'], original, 128, 0, 1, 1, 896, **options)
        stage('global-prepare'); runner.launch('prepare', nonblock=False)
        normal['ready'] = read('progress', 0, 0, 129, 20, 1)
        np.savez(directory / 'normal-d2h.npz', **normal)
        np.testing.assert_array_equal(normal['ready'], np.ones(2580, np.uint32))
        stage('compute-device-chain'); runner.launch('compute', nonblock=False)
        normal['output'] = read('output', 128, 0, 1, 1, 896)
        np.savez(directory / 'normal-d2h.npz', **normal)
        normal['progress'] = read('progress', 0, 0, 129, 20, 1)
        np.savez(directory / 'normal-d2h.npz', **normal)
        np.testing.assert_array_equal(normal['progress'], np.full(2580, 2, np.uint32))
    finally:
        try:
            stage('stop'); runner.stop()
        finally:
            # Bind existing artifacts even if stop itself fails. Their presence
            # does not imply successful compute, drain or complete core output.
            (directory / 'runtime-evidence.json').write_text(json.dumps(dict(
                completed_host_reads=sorted(normal),
                files={name: sha(directory / name) for name in ('out.core', 'normal-d2h.npz')
                       if (directory / name).is_file()},
                compiled_files_sha256=sha(root / 'compiled-files.json')),
                indent=2)+'\n')
    stage('verify-runtime-artifacts')
    verify(root)
    verify_images()
    core_path = directory / 'out.core'
    core = StoppedCore.from_verified_contract(core_path,
        json.loads((root / 'assembled/symbols.json').read_text()), assembly['composition'])
    assembly_config = json.loads((root / 'assembled/config.json').read_text())
    compiled_roots = []
    for row in assembly_config['audits']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        compiled_roots.append(root / 'assembled' / relative / 'compiled')
    expected_initial = complete_expected(compiled_roots)
    stage('validate-actual-core')
    result = validate(core, expected_initial, reference, normal, policy, directory)
    (root / 'results.json').write_text(json.dumps(dict(success=True, validation=result,
        assembly_execution_sha256=config['assembly_execution_sha256'],
        reference_inputs_sha256=REFERENCE_INPUTS, compiled_files_sha256=sha(root / 'compiled-files.json'),
        output_sha256={name: sha(directory / name) for name in ('actual.npz', 'protocol.json', 'normal-d2h.npz', 'out.core')},
        scope=config['scope']), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--assembly', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.assembly:
        if args.reference is None:
            parser.error('--reference is required')
        prepare(args.assembly.resolve(), args.reference.resolve(), args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 1800, rss_limit_kib=8*1024*1024)

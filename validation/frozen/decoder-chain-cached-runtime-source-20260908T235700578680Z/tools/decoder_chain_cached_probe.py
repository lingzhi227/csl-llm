"""One persistent SDK instance: first token, cached token, reset repeat.

Use the accepted original two-layer assembly unchanged. Host transfers only
original embedding inputs; no host neural computation or interlayer relay.
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


def cached_receipt(build):
    from sdk_probe import verify, sha
    verify(build)
    execution = json.loads((build / 'execution.json').read_text())
    result = json.loads((build / 'results.json').read_text())
    assert execution['success'] and execution['cpu_only'] and result['success'] and result['cpu_only']
    assert execution['manifest_sha256'] == sha(build / 'manifest.json')
    assert execution['results_sha256'] == sha(build / 'results.json')
    return result


def prepare(assembled, original, build, source, project):
    sys.path[:0] = [str(source / 'tools'), str(source / 'src')]
    from sdk_probe import verify, sha
    from sdk_core import contract_symbols
    from decoder_chain_runtime_probe import assembled_receipt, REFERENCE_MANIFEST, REFERENCE_INPUTS
    from decoder_chain_cached_validate import case_plan
    manifest, assembly = assembled_receipt(assembled)
    contract_symbols(json.loads((assembled / 'symbols.json').read_text()), assembly['composition'])
    verify(original)
    assert sha(original / 'manifest.json') == REFERENCE_MANIFEST
    assert sha(original / 'inputs.npz') == REFERENCE_INPUTS
    receipt = cached_receipt(build)
    cached = Path(receipt['reference_bundle']).resolve()
    if cached.parent != project / 'evidence' or not cached.name.startswith('decoder-chain-cached-reference-'):
        raise ValueError('Expected a local immutable cached reference artifact')
    verify(cached)
    assert sha(cached / 'manifest.json') == receipt['reference_manifest_sha256']
    assert sha(cached / 'inputs.npz') == receipt['reference_inputs_sha256']
    cached_config = json.loads((cached / 'config.json').read_text())
    original_config = json.loads((original / 'config.json').read_text())
    assert cached_config['prefix_length'] == 2 and cached_config['positions'] == [0, 1]
    assert cached_config['layers'] == [0, 1] and len(cached_config['token_ids']) == 2
    assert cached_config['token_ids'][0] == original_config['token_id'] == 151644
    for key in ('model_sha256', 'model_manifest_sha256', 'official_report_sha256',
                'official_trace_sha256', 'frequency_auxiliary_sha256'):
        assert cached_config[key] == original_config[key]
    out = project / 'evidence' / ('decoder-chain-cached-runtime-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    for path, name in ((assembled, 'assembled'), (original, 'reference'),
                       (build, 'reference-build'), (cached, 'cached-reference')):
        shutil.copytree(path, out / name, ignore=shutil.ignore_patterns('__pycache__'))
    names = ['tools/decoder_chain_runtime_probe.py', 'tools/decoder_chain_cached_validate.py',
             'tools/decoder_chain_validate.py', 'tools/decoder_chain_symbols.py',
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
    assert sha(out / 'precision.json') == sha(original / 'precision.json') == sha(cached / 'precision.json')
    (out / 'config.json').write_text(json.dumps(dict(
        assembly_execution_sha256=sha(assembled / 'execution.json'),
        reference_build_execution_sha256=sha(build / 'execution.json'),
        original_reference_manifest_sha256=REFERENCE_MANIFEST,
        original_reference_inputs_sha256=REFERENCE_INPUTS,
        cached_reference_manifest_sha256=receipt['reference_manifest_sha256'],
        cached_reference_inputs_sha256=receipt['reference_inputs_sha256'],
        cases=case_plan(), token_ids=cached_config['token_ids'], timeout_seconds=3600,
        rss_limit_kib=8*1024*1024, simulator_threads=8,
        scope='Three calls in one original two-layer SDK instance: position0, cached position1, reset position0. '
              'Per-call normal outputs/status and final actual-core detailed checks. '
              'No long-context, forced arrival ordering, full-model or continuous generation acceptance.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    sys.path[:0] = [str(root / 'tools'), str(root / 'src')]
    from executor import verify, sha
    from sdk_core import StoppedCore
    from decoder_chain_runtime_probe import assembled_receipt, REFERENCE_MANIFEST, REFERENCE_INPUTS
    from decoder_chain_assemble import complete_expected
    from decoder_chain_validate import validate
    from decoder_chain_cached_validate import case_plan, validate_case, validate_repeat
    from csl_llm.decoder_layout import geometry
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    verify(root)
    config = json.loads((root / 'config.json').read_text())
    assert (config['timeout_seconds'], config['rss_limit_kib'], config['simulator_threads']) == (3600, 8*1024*1024, 8)
    assert config['cases'] == case_plan()
    assert sha(root / 'assembled/execution.json') == config['assembly_execution_sha256']
    _, assembly = assembled_receipt(root / 'assembled')
    assert sha(root / 'reference-build/execution.json') == config['reference_build_execution_sha256']
    receipt = cached_receipt(root / 'reference-build')
    assert sha(root / 'reference/manifest.json') == config['original_reference_manifest_sha256'] == REFERENCE_MANIFEST
    assert sha(root / 'reference/inputs.npz') == config['original_reference_inputs_sha256'] == REFERENCE_INPUTS
    assert sha(root / 'cached-reference/manifest.json') == config['cached_reference_manifest_sha256'] == receipt['reference_manifest_sha256']
    assert sha(root / 'cached-reference/inputs.npz') == config['cached_reference_inputs_sha256'] == receipt['reference_inputs_sha256']
    original, cached = np.load(root / 'reference/inputs.npz'), np.load(root / 'cached-reference/inputs.npz')
    assert cached['input'].dtype == np.float32 and cached['input'].shape == (2, 896)
    assert np.isfinite(cached['input']).all() and cached['input'][:1].tobytes() == original['input'].tobytes()
    policy = json.loads((root / 'precision.json').read_text())['stage_gate']['absolute_or_normwise']
    assert policy == dict(max_abs=1e-5, both=dict(relative_l2=2e-4, relative_peak=5e-4))
    directory = root / 'runtime'; directory.mkdir()
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
    ids = {name: runner.get_id(name) for name in ('input', 'output', 'progress', 'status', 'timing')}
    options = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_32BIT,
                   order=MemcpyOrder.ROW_MAJOR, nonblock=False)
    def read(name, x, y, width, height, count):
        stage('d2h-'+name)
        values = np.zeros(width*height*count, np.float32 if name == 'output' else np.uint32)
        settings = dict(options)
        if name == 'timing': settings['data_type'] = MemcpyDataType.MEMCPY_16BIT
        runner.memcpy_d2h(values, ids[name], x, y, width, height, count, **settings)
        return values
    normal, reports = [], []
    def save_normal():
        np.savez(directory / 'normal-d2h.npz', **{f'case_{i}_{name}': value
            for i, case in enumerate(normal) for name, value in case.items()})
    try:
        stage('load'); runner.load()
        stage('run'); runner.run()
        stage('initialize'); runner.launch('initialize', nonblock=False)
        for index, plan in enumerate(config['cases']):
            actual = {}; normal.append(actual)
            stage(plan['label']+'-begin')
            if plan['reset']:
                stage('reset-model'); runner.launch('reset_model', nonblock=False)
            stage('initial-embedding-h2d')
            runner.memcpy_h2d(ids['input'], np.ascontiguousarray(cached['input'][plan['input_index']]),
                              128, 0, 1, 1, 896, **options)
            stage('global-prepare'); runner.launch('prepare', nonblock=False)
            actual['ready'] = read('progress', 0, 0, 129, 20, 1); save_normal()
            np.testing.assert_array_equal(actual['ready'], np.full(2580, plan['invocation']*2-1, np.uint32))
            stage('compute-device-chain'); runner.launch('compute', nonblock=False)
            actual['output'] = read('output', 128, 0, 1, 1, 896); save_normal()
            actual['progress'] = read('progress', 0, 0, 129, 20, 1); save_normal()
            np.testing.assert_array_equal(actual['progress'], np.full(2580, plan['invocation']*2, np.uint32))
            actual['endpoint_timing'] = read('timing', 128, 0, 1, 1, 6)
            for layer in (0, 1):
                x = layer*64+62
                actual[f'layer_{layer}_output'] = read('output', x, 0, 1, 1, 896)
                actual[f'controller_status_{layer}'] = read('status', x, 0, 1, 1, 2)
                actual[f'layer_{layer}_timing'] = read('timing', x, 0, 1, 1, 6)
                for head, region in enumerate(geometry((layer*64, 0))['kv']):
                    actual[f'kv_status_{layer}_{head}'] = read('status', region.x, region.y, 11, 6, 2).reshape(66, 2)
            save_normal()
            reports.append(validate_case(actual, index, original, cached, policy))
            (directory / 'case-reports.json').write_text(json.dumps(dict(completed_cases=reports,
                scope='Intermediate normal readbacks only; final core checks remain required.'), indent=2)+'\n')
        repeated = validate_repeat(normal)
    finally:
        try:
            stage('stop'); runner.stop()
        finally:
            (directory / 'runtime-evidence.json').write_text(json.dumps(dict(
                completed_host_reads=[sorted(case) for case in normal],
                files={name: sha(directory / name) for name in ('out.core', 'normal-d2h.npz', 'case-reports.json')
                       if (directory / name).is_file()},
                compiled_files_sha256=sha(root / 'compiled-files.json')), indent=2)+'\n')
    stage('verify-runtime-artifacts'); verify(root); verify_images()
    core = StoppedCore.from_verified_contract(directory / 'out.core',
        json.loads((root / 'assembled/symbols.json').read_text()), assembly['composition'])
    assembly_config = json.loads((root / 'assembled/config.json').read_text())
    compiled_roots = []
    for row in assembly_config['audits']:
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        compiled_roots.append(root / 'assembled' / relative / 'compiled')
    expected = complete_expected(compiled_roots)
    stage('validate-final-reset-core')
    final = validate(core, expected, original, normal[-1], policy, directory,
                     invocation=3, retained_cache=cached)
    # Normal 16-bit SDK reads are held in u32 words; final core stores u16.
    final_arrays = np.load(directory / 'actual.npz')
    for layer in (0, 1):
        assert final_arrays[f'layer_{layer}_output'].tobytes() == normal[-1][f'layer_{layer}_output'].tobytes()
    for name in ('endpoint_timing', 'layer_0_timing', 'layer_1_timing'):
        assert final_arrays[name].tobytes() == normal[-1][name].astype('<u2').tobytes()
    (root / 'results.json').write_text(json.dumps(dict(success=True, cases=reports,
        reset_repeat=repeated, final_core=final,
        assembly_execution_sha256=config['assembly_execution_sha256'],
        reference_build_execution_sha256=config['reference_build_execution_sha256'],
        compiled_files_sha256=sha(root / 'compiled-files.json'),
        output_sha256={name: sha(directory / name) for name in ('actual.npz', 'protocol.json',
            'normal-d2h.npz', 'case-reports.json', 'runtime-evidence.json', 'out.core')},
        scope=config['scope']), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--assembly', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--cached-build', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.assembly:
        if args.reference is None or args.cached_build is None:
            parser.error('--reference and --cached-build are required')
        prepare(args.assembly.resolve(), args.reference.resolve(), args.cached_build.resolve(),
                args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 3600, rss_limit_kib=8*1024*1024)

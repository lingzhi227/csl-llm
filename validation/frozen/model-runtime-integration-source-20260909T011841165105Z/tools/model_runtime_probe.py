"""Guarded first complete original-model SDK integration, one token and head.

No complete assembly exists merely because this driver exists. Preparation
requires all original partitions and that exact artifact's diagnostic symbols.
An explicit reviewed wall-time budget is required; no model timing is inferred
from the two-layer gate or the bootstrap. This is not S5/S6 completion.
"""
import argparse
import datetime
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_MANIFEST = '89d1f64693e8dd641421a36285c0a74e040f6b1bfb227d10089cddb23c9a7b4c'
REFERENCE_REPORT = '8c3c1349f46c0359e20088df597b9a1e25ce1053613aeeeeab68edc9151bf23f'
MODEL_MANIFEST = '9e4305503478f0f118ced0c5b87b148b73ae5823504f5636fb7c778e1252c75e'
PACK_MANIFEST = '788ce538ac7ca41f2547b4ce8b86bee283b2cbd75b739f3841fdafafca9e0c9b'


def assembled_receipt(root):
    from sdk_probe import verify, sha
    from sdk_core import contract_symbols
    from model_partition_assemble import complete_coverage
    manifest = verify(root)
    execution = json.loads((root/'execution.json').read_text())
    result = json.loads((root/'results.json').read_text())
    assert execution['success'] and result['success'] and result['passed'] and result['offline_only']
    assert execution['manifest_sha256'] == sha(root/'manifest.json')
    assert execution['results_sha256'] == sha(root/'results.json')
    assert execution['sdk_sha256'] == manifest['sdk_sha256']
    assert result['original_matrix_tiles'] == 34552 and result['application_coordinates'] == 40530
    config = json.loads((root/'config.json').read_text())
    roots, inputs = [], []
    assert len(config['audits']) == len(result['contracts'])
    for row, contract in zip(config['audits'], result['contracts']):
        relative = Path(row['path'])
        assert not relative.is_absolute() and '..' not in relative.parts
        path = root/relative
        assert sha(path/'execution.json') == row['execution_sha256'] == contract['source_execution_sha256']
        assert sha(path/'initializers.json') == contract['initializer_report_sha256']
        original = json.loads((path/'compiled/static-input-config.json').read_text())
        assert original['mapping_audit']['pack_manifest_sha256'] == PACK_MANIFEST
        inputs.append(original)
        roots.append(path/'compiled')
    complete_coverage(inputs)
    composition = result['composition']
    assert composition['passed'] and composition['coordinate_count'] == 40530
    names = {str(p.relative_to(root/'out')) for p in (root/'out').rglob('*') if p.is_file()}
    assert names == set(composition['artifact_sha256'])
    for name, expected in composition['artifact_sha256'].items():
        relative = Path(name)
        assert not relative.is_absolute() and '..' not in relative.parts
        assert sha(root/'out'/relative) == expected
    assert sha(root/'symbols.json') == result['symbols_sha256']
    symbols = json.loads((root/'symbols.json').read_text())
    assert symbols['diagnostics'] is True
    contract_symbols(symbols, composition)
    return manifest, result, roots


def reference_receipt(build, reference):
    from sdk_probe import verify, sha
    verify(build); verify(reference)
    config = json.loads((build/'config.json').read_text())
    execution = json.loads((build/'execution.json').read_text())
    result = json.loads((build/'results.json').read_text())
    assert config['reference_kind'] == 'model-single-token'
    assert execution['success'] and execution['cpu_only'] and result['success'] and result['cpu_only']
    assert execution['manifest_sha256'] == sha(build/'manifest.json')
    assert execution['results_sha256'] == sha(build/'results.json')
    assert sha(reference/'manifest.json') == result['reference_manifest_sha256'] == REFERENCE_MANIFEST
    assert sha(reference/'report.json') == result['reference_report_sha256'] == REFERENCE_REPORT
    assert sha(reference/'single-token/step-000.npz') == result['reference_trace_sha256']


def prepare(assembled, reference, build, source, project, timeout_seconds):
    if type(timeout_seconds) is not int or not 1800 <= timeout_seconds <= 86400:
        raise ValueError('Declare a reviewed first-model measurement budget in [1800,86400] seconds')
    sys.path[:0] = [str(source/'tools'), str(source/'src')]
    from sdk_probe import sha
    manifest, _, _ = assembled_receipt(assembled)
    reference_receipt(build, reference)
    out = project/'evidence'/('model-runtime-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    for path, label in [(assembled, 'assembled'), (reference, 'reference'), (build, 'reference-build')]:
        shutil.copytree(path, out/label, ignore=shutil.ignore_patterns('__pycache__'))
    names = ['tools/sdk_probe.py', 'tools/sdk_core.py', 'tools/model_reference.py',
        'tools/model_step_validate.py', 'tools/model_runtime_io.py', 'tools/model_core_validate.py',
        'tools/model_partition_assemble.py', 'tools/model_initializer_audit.py',
        'tools/model_symbols.py', 'tools/decoder_chain_symbols.py',
        'src/csl_llm/__init__.py', 'src/csl_llm/coordinate_identity.py',
        'src/csl_llm/decoder_layout.py', 'src/csl_llm/regions.py']
    for name in names:
        path = out/name; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, path)
    shutil.copy2(Path(__file__), out/'driver.py')
    shutil.copy2(source/'tools/sdk_probe.py', out/'executor.py')
    shutil.copy2(source/'configs/precision.json', out/'precision.json')
    assert sha(out/'precision.json') == sha(reference/'precision.json')
    (out/'config.json').write_text(json.dumps(dict(
        assembly_execution_sha256=sha(assembled/'execution.json'),
        reference_build_execution_sha256=sha(build/'execution.json'),
        timeout_seconds=timeout_seconds, rss_limit_kib=24*1024*1024, simulator_threads=16,
        prompt=[151644], generation_limit=1,
        scope='First single-token original24layer/full151936 device head integration. '
              'One raw token151644, original resident weights, device embedding/finalRMS/greedy. '
              'Not complete fixed prompts, cached generation, reset, long context, S5/S6 or hardware acceptance.'), indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    import numpy as np
    sys.path[:0] = [str(root/'tools'), str(root/'src')]
    from executor import verify, sha
    from sdk_core import StoppedCore
    from model_reference import OfficialModelReference
    from model_step_validate import validate_step
    from model_runtime_io import ModelRuntimeIO
    from model_core_validate import validate as validate_core
    from model_partition_assemble import iter_initializers
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    verify(root)
    config = json.loads((root/'config.json').read_text())
    assert config['prompt'] == [151644] and config['generation_limit'] == 1
    assert (config['rss_limit_kib'], config['simulator_threads']) == (24*1024*1024, 16)
    assert type(config['timeout_seconds']) is int and 1800 <= config['timeout_seconds'] <= 86400
    assert sha(root/'assembled/execution.json') == config['assembly_execution_sha256']
    assert sha(root/'reference-build/execution.json') == config['reference_build_execution_sha256']
    _, assembly, compiled_roots = assembled_receipt(root/'assembled')
    reference_receipt(root/'reference-build', root/'reference')
    reference = OfficialModelReference(root/'reference', report_sha256=REFERENCE_REPORT,
        model_manifest_sha256=MODEL_MANIFEST, prompt_id='single-token', limit=1)
    assert reference.prompt == config['prompt'] and len(reference.steps) == 1
    expected, plan = reference.step(0), reference.steps[0]
    from model_reference import DECODER_STAGES
    assert all(f'layer_{layer}_{name}' in expected for layer in range(24) for name in DECODER_STAGES)
    precision = json.loads((root/'precision.json').read_text())
    assert precision['stage_gate']['absolute_or_normwise'] == dict(max_abs=1e-5, both=dict(relative_l2=2e-4, relative_peak=5e-4))
    assert precision['logit_gate']['all_required'] == dict(max_abs=.01, relative_l2=2e-4, relative_peak=5e-4)
    directory = root/'runtime'; directory.mkdir()
    shutil.copytree(root/'assembled/out', directory/'out')
    artifacts = assembly['composition']['artifact_sha256']
    (root/'compiled-files.json').write_text(json.dumps(artifacts, indent=2)+'\n')
    start = time.monotonic()
    def stage(name):
        event = dict(stage=name, elapsed_seconds=time.monotonic()-start)
        (root/'stage.json').write_text(json.dumps(event)+'\n')
        with (root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(name, flush=True)
    def save(record):
        np.savez(directory/'normal-d2h.npz', **record)
    os.chdir(directory)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True,
        num_threads=16, dump_core=True), SdkTarget.WSE3))
    io = None
    try:
        stage('load'); runner.load()
        stage('run'); runner.run()
        stage('initialize'); runner.launch('initialize', nonblock=False)
        io = ModelRuntimeIO(runner, memcpy_data_type=MemcpyDataType,
            memcpy_order=MemcpyOrder, save=save, stage=stage)
        io.reset_request(reference.prompt, 1)
        actual = io.step(head=True, generated_count=1)
        normal_report = validate_step(actual, expected, plan, invocation=1,
            prompt=reference.prompt, limit=1, precision=precision)
        (directory/'normal-report.json').write_text(json.dumps(normal_report, indent=2)+'\n')
        io.accept_step()
    finally:
        try:
            stage('stop'); runner.stop()
        finally:
            (directory/'runtime-evidence.json').write_text(json.dumps(dict(
                files={name: sha(directory/name) for name in ('out.core', 'normal-d2h.npz', 'normal-report.json')
                       if (directory/name).is_file()}, compiled_files_sha256=sha(root/'compiled-files.json')), indent=2)+'\n')
    # Release the simulator instance before loading a full-fabric actual core.
    del io, runner
    gc.collect()
    stage('verify-runtime-artifacts'); verify(root)
    for name, digest in artifacts.items():
        assert sha(directory/'out'/name) == digest
    stage('actual-full-model-core')
    core = StoppedCore.from_verified_contract(directory/'out.core',
        json.loads((root/'assembled/symbols.json').read_text()), assembly['composition'])
    core_report = validate_core(core, iter_initializers(compiled_roots), actual, [expected],
        invocation=1, precision=precision, output=directory)
    reference.close()
    (root/'results.json').write_text(json.dumps(dict(success=True,
        normal=normal_report, core=core_report,
        assembly_execution_sha256=config['assembly_execution_sha256'],
        reference_build_execution_sha256=config['reference_build_execution_sha256'],
        output_sha256={name: sha(directory/name) for name in ('normal-d2h.npz', 'normal-report.json',
            'actual.npz', 'protocol.json', 'runtime-evidence.json', 'out.core')}, scope=config['scope']), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--assembly', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--reference-build', type=Path)
    parser.add_argument('--timeout-seconds', type=int)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.assembly:
        if args.reference is None or args.reference_build is None or args.timeout_seconds is None:
            parser.error('--reference, --reference-build and explicit --timeout-seconds are required')
        prepare(args.assembly.resolve(), args.reference.resolve(), args.reference_build.resolve(),
                args.source_root.resolve(), args.project_root.resolve(), args.timeout_seconds)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        root = args.execute.resolve()
        sys.path.insert(0, str(root))
        from executor import verify, execute
        verify(root)
        config = json.loads((root/'config.json').read_text())
        assert type(config['timeout_seconds']) is int and 1800 <= config['timeout_seconds'] <= 86400
        assert config['rss_limit_kib'] == 24*1024*1024
        execute(root, config['timeout_seconds'], rss_limit_kib=24*1024*1024)

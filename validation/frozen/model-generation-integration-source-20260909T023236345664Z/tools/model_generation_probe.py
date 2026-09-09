"""Guarded persistent fixed-corpus candidate, after actual single-token success.

Source authoring only until complete original assembly and the first actual
model integration are available. Preparing this candidate does not authorize
its execution or establish long-context, capacity or hardware acceptance.
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


def limits(config):
    seconds = config['timeout_seconds']
    if (type(seconds) is not int or not 1800 <= seconds <= 86400 or
            type(config['rss_limit_kib']) is not int or type(config['simulator_threads']) is not int or
            config['rss_limit_kib'] != 24*1024*1024 or config['simulator_threads'] != 16):
        raise ValueError('Explicit reviewed generation budget and fixed24GiB/16thread guard required')
    return seconds


def first_admission_receipt(root, config):
    """Recheck metadata copied only after full actual first-run admission."""
    from sdk_probe import sha
    proof = json.loads((root/'first-integration/admission.json').read_text())
    if sha(root/'first-integration/admission.json') != config['first_admission_sha256']:
        raise ValueError('Changed first-integration admission')
    for name in ('manifest', 'execution', 'results'):
        if sha(root/f'first-integration/{name}.json') != proof[name+'_sha256']:
            raise ValueError('Changed actual first-integration receipt')
    execution = json.loads((root/'first-integration/execution.json').read_text())
    result = json.loads((root/'first-integration/results.json').read_text())
    if (proof['assembly_execution_sha256'] != config['assembly_execution_sha256'] or
            execution['manifest_sha256'] != proof['manifest_sha256'] or
            execution['results_sha256'] != proof['results_sha256'] or
            execution['sdk_sha256'] != proof['sdk_sha256'] or
            execution['success'] is not True or result['success'] is not True or
            result['output_sha256'] != proof['actual_output_sha256']):
        raise ValueError('First-integration metadata does not bind this original assembly')
    return proof


def prepare(assembled, first, reference, source, project, timeout_seconds):
    sys.path[:0] = [str(source/'tools'), str(source/'src')]
    from sdk_probe import verify, sha
    from model_runtime_probe import assembled_receipt
    from model_integration_admission import first_single_token
    from model_fixed_requests import open_references, REPORT_SHA256
    config = dict(timeout_seconds=timeout_seconds, rss_limit_kib=24*1024*1024,
                  simulator_threads=16)
    limits(config)
    source_manifest = verify(source)
    if sha(source/'tools/model_generation_probe.py') != sha(Path(__file__)):
        raise ValueError('Prepare must use the selected frozen generation driver')
    manifest, _, _ = assembled_receipt(assembled)
    config['assembly_execution_sha256'] = sha(assembled/'execution.json')
    proof = first_single_token(first, assembly_execution_sha256=config['assembly_execution_sha256'],
                               sdk_sha256=manifest['sdk_sha256'])
    refs = open_references(reference)
    for item in refs:
        item.close()
    out = project/'evidence'/('model-generation-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    shutil.copytree(assembled, out/'assembled', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(reference, out/'reference', ignore=shutil.ignore_patterns('__pycache__'))
    for name in source_manifest['files']:
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Invalid frozen source path')
        path = out/'source'/relative; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/relative, path)
    shutil.copy2(source/'manifest.json', out/'source/manifest.json')
    first_copy = out/'first-integration'; first_copy.mkdir()
    for name in ('manifest.json', 'execution.json', 'results.json', 'config.json'):
        shutil.copy2(first/name, first_copy/name)
    (first_copy/'admission.json').write_text(json.dumps(proof, indent=2)+'\n')
    config.update(first_admission_sha256=sha(first_copy/'admission.json'),
        reference_report_sha256=REPORT_SHA256,
        source_manifest_sha256=sha(source/'manifest.json'),
        scope='Four persistent fixed requests: arithmetic, English, Chinese, arithmetic reset-repeat; '
              'original24layers/full151936 and original resident weights. '
              'Not2048context/256generation capacity or hardware qualification.')
    shutil.copy2(Path(__file__), out/'driver.py')
    shutil.copy2(source/'tools/sdk_probe.py', out/'executor.py')
    shutil.copy2(source/'configs/precision.json', out/'precision.json')
    (out/'config.json').write_text(json.dumps(config, indent=2)+'\n')
    (out/'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    sys.path[:0] = [str(root/'source/tools'), str(root/'source/src')]
    from sdk_probe import verify, sha
    from sdk_core import StoppedCore
    from model_runtime_probe import assembled_receipt
    from model_generation_runtime import run
    from model_core_validate import validate as validate_core
    from model_partition_assemble import iter_initializers
    from model_fixed_requests import REPORT_SHA256
    from cerebras.sdk.runtime.sdkruntimepybind import (
        SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
    manifest = verify(root)
    config = json.loads((root/'config.json').read_text()); limits(config)
    if (sha(root/'assembled/execution.json') != config['assembly_execution_sha256'] or
            sha(root/'source/manifest.json') != config['source_manifest_sha256'] or
            config['reference_report_sha256'] != REPORT_SHA256):
        raise ValueError('Generation provenance changed')
    proof = first_admission_receipt(root, config)
    if proof['sdk_sha256'] != manifest['sdk_sha256']:
        raise ValueError('First integration used a different SDK')
    _, assembled, roots = assembled_receipt(root/'assembled')
    precision = json.loads((root/'precision.json').read_text())
    if (precision['stage_gate']['absolute_or_normwise'] != dict(max_abs=1e-5,
            both=dict(relative_l2=2e-4, relative_peak=5e-4)) or
            precision['logit_gate']['all_required'] != dict(max_abs=.01, relative_l2=2e-4, relative_peak=5e-4)):
        raise ValueError('Frozen numerical gates changed')
    directory = root/'runtime'; directory.mkdir()
    shutil.copytree(root/'assembled/out', directory/'out')
    start = time.monotonic()
    def stage(name):
        event = dict(stage=name, elapsed_seconds=time.monotonic()-start)
        (root/'stage.json').write_text(json.dumps(event)+'\n')
        with (root/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(event)+'\n')
        print(name, flush=True)
    os.chdir(directory)
    runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True,
        num_threads=16, dump_core=True), SdkTarget.WSE3))
    outcome = run(runner, directory=directory, reference_directory=root/'reference', precision=precision,
        memcpy_data_type=MemcpyDataType, memcpy_order=MemcpyOrder, stage=stage)
    del runner
    gc.collect()
    stage('verify-runtime-artifacts'); verify(root)
    artifacts = assembled['composition']['artifact_sha256']
    if {str(p.relative_to(directory/'out')) for p in (directory/'out').rglob('*') if p.is_file()} != set(artifacts):
        raise ValueError('Runtime artifact file set differs from the original assembly')
    for name, digest in artifacts.items():
        if sha(directory/'out'/name) != digest:
            raise ValueError('Runtime changed a compiled original artifact')
    stage('actual-full-model-core')
    core = StoppedCore.from_verified_contract(directory/'out.core',
        json.loads((root/'assembled/symbols.json').read_text()), assembled['composition'])
    report = validate_core(core, iter_initializers(roots), outcome['normal'], outcome['reference_history'],
        invocation=outcome['invocation'], precision=precision, output=directory)
    (root/'results.json').write_text(json.dumps(dict(success=True, core=report,
        requests=outcome['requests'], invocation=outcome['invocation'], reset_repeat_bit_exact=True,
        assembly_execution_sha256=config['assembly_execution_sha256'],
        first_admission_sha256=config['first_admission_sha256'],
        output_sha256={str(p.relative_to(directory)): sha(p) for p in directory.rglob('*')
                      if p.is_file() and 'out' not in p.relative_to(directory).parts},
        scope=config['scope']), indent=2)+'\n')
    stage('complete')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--assembly', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--first-integration', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--source-root', type=Path)
    parser.add_argument('--project-root', type=Path)
    parser.add_argument('--timeout-seconds', type=int)
    args = parser.parse_args()
    if args.assembly:
        if any(value is None for value in (args.first_integration, args.reference, args.source_root,
                                           args.project_root, args.timeout_seconds)):
            parser.error('Preparation requires first integration, reference, frozen source, project and explicit budget')
        prepare(args.assembly.resolve(), args.first_integration.resolve(), args.reference.resolve(),
                args.source_root.resolve(), args.project_root.resolve(), args.timeout_seconds)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        root = args.execute.resolve()
        sys.path.insert(0, str(root))
        from executor import verify, execute
        verify(root)
        config = json.loads((root/'config.json').read_text())
        execute(root, limits(config), rss_limit_kib=config['rss_limit_kib'])

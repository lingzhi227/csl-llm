"""Freeze one bounded original-weight partition of the native two-layer gate."""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
ROOT = Path(__file__).resolve().parents[1]

DEVICE_FILES = {
    'decoder.csl': 'csl/programs/resident_decoder/pe.csl',
    'chain_endpoint.csl': 'csl/programs/decoder_chain/endpoint.csl',
    'kernel.csl': 'csl/kernels/local_gemv_bf16_f32_colmajor.csl',
    'vector.csl': 'csl/kernels/vector_f32.csl',
    'rope.csl': 'csl/kernels/qwen_position_f32.csl',
    'kv.csl': 'csl/kernels/kv_block_f32.csl',
    'line.csl': 'csl/runtime/line_allreduce_f32.csl',
    'input.csl': 'csl/runtime/filtered_input.csl',
    'packets.csl': 'csl/runtime/packets.csl',
    'activation_stream.csl': 'csl/runtime/activation_stream.csl',
}


def prepare(static_inputs, source, project):
    sys.path.insert(0, str(source / 'tools'))
    sys.path.insert(0, str(source / 'src'))
    from sdk_probe import sha
    from model_static_layout import load_bindings
    from decoder_chain_layout import generate
    data, files, binding = load_bindings(static_inputs)
    config = json.loads((static_inputs / 'config.json').read_text())
    for row in config['matrices']:
        x, y = row['coordinate']
        original = row['original_record']
        if not (original['layer'] in (0, 1) and 0 <= x < 128 and 0 <= y < 20):
            raise ValueError('Only original layers 0 and 1 belong to this gate')
    local = {point: value for point, value in data.items() if point[0] < 128 and point[1] < 20}
    layout = generate(local)
    binding = dict(binding, initialized_controllers=2,
                   scope='Original bounded matrix selection and two original norm/frequency controllers; compile-only.')
    out = project / 'evidence' / ('decoder-chain-compile-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    for target, relative in DEVICE_FILES.items():
        shutil.copy2(source / relative, out / target)
    for name in files:
        if (out / name).exists():
            raise ValueError('Initializer collides with program source')
        shutil.copy2(static_inputs / name, out / name)
    for name, target in [('manifest.json', 'static-input-manifest.json'), ('config.json', 'static-input-config.json')]:
        shutil.copy2(static_inputs / name, out / target)
    builders = ['tools/decoder_chain_layout.py', 'tools/resident_decoder_probe.py',
                'tools/model_static_layout.py', 'tools/audit_model_weight_mapping.py', 'tools/verify_weight_pack.py',
                'src/csl_llm/__init__.py', 'src/csl_llm/decoder_layout.py', 'src/csl_llm/regions.py', 'src/csl_llm/static_data.py']
    for name in builders:
        target = out / 'builder' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / name, target)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    shutil.copy2(source / 'configs/precision.json', out / 'precision.json')
    (out / 'layout.csl').write_text(layout)
    (out / 'config.json').write_text(json.dumps(dict(width=129, height=20,
        fabric_dimensions=[136, 22], fabric_offsets=[4, 1], layers=[0, 1],
        matrix_tiles=2088, kv_tiles=264, controllers=2, endpoint=[128, 0],
        weight_storage='u8', static_input_binding=binding,
        timeout_seconds=900, rss_limit_kib=12*1024*1024,
        production_source_sha256={relative: sha(source / relative) for target, relative in DEVICE_FILES.items()
                                  if target != 'chain_endpoint.csl'},
        scope='Two original-sized streamed decoder regions plus diagnostic endpoint. '
              'Selected original matrices and both norm/frequency controllers, selected biases; '
              'remaining matrices/biases zero. Compilation only, no complete initialization or SDK inference.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
        files={str(p.relative_to(out)): sha(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    from executor import verify
    verify(root)
    os.chdir(root)
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=136,22',
               '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1',
               '--max-parallelism=1', '--dump-dsr-alloc-graph']
    (root / 'compile-command.json').write_text(json.dumps(command)+'\n')
    subprocess.run(command, check=True)
    images = list((root / 'out/bin').glob('*.elf'))
    assert images
    (root / 'results.json').write_text(json.dumps(dict(success=True, compile_only=True,
        elf_count=len(images), scope=json.loads((root / 'config.json').read_text())['scope']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--static-inputs', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.static_inputs:
        prepare(args.static_inputs.resolve(), args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 900, rss_limit_kib=12*1024*1024)

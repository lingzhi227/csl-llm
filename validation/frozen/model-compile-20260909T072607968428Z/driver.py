"""Freeze and compile the complete resident model CSL graph.

This is a compiler/resource check, never an initialized inference acceptance.
All original matrix/KV owners are present. Initial data are either zero or a
bounded selection of original declarations; no SdkRuntime is constructed.
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

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def resource_limits(rss_limit_gib):
    if type(rss_limit_gib) is not int or rss_limit_gib not in (8, 12):
        raise ValueError('Compile RSS guard must be explicitly 8 or 12 GiB')
    return dict(timeout_seconds=900, rss_limit_kib=rss_limit_gib*1024*1024)


def prepare(source, project, weight_storage='u32', static_inputs=None, rss_limit_gib=8):
    limits = resource_limits(rss_limit_gib)
    sys.path.insert(0, str(source / 'tools'))
    import model_layout
    data = None
    data_files = []
    binding_report = None
    if static_inputs is not None:
        if weight_storage != 'u8':
            raise ValueError('Native static declarations require u8 weight storage')
        from model_static_layout import load_bindings
        data, data_files, binding_report = load_bindings(static_inputs)
    out = project / 'evidence' / ('model-compile-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    (out / 'layout.csl').write_text(model_layout.generate(weight_storage=weight_storage, static_data=data))
    files = [(Path(__file__), 'driver.py'),
             (source / 'tools/sdk_probe.py', 'executor.py'),
             (source / 'csl/programs/resident_decoder/pe.csl', 'decoder.csl'),
             (source / 'csl/programs/model_vocabulary/pe.csl', 'vocabulary.csl')]
    files += [(source / 'csl' / directory / name, target) for directory, name, target in [
        ('kernels', 'local_gemv_bf16_f32_colmajor.csl', 'kernel.csl'),
        ('kernels', 'vector_f32.csl', 'vector.csl'),
        ('kernels', 'qwen_position_f32.csl', 'rope.csl'),
        ('kernels', 'kv_block_f32.csl', 'kv.csl'),
        ('kernels', 'tied_embedding_f32.csl', 'embedding.csl'),
        ('kernels', 'argmax_f32.csl', 'argmax.csl'),
        ('runtime', 'line_allreduce_f32.csl', 'line.csl'),
        ('runtime', 'filtered_input.csl', 'input.csl'),
        ('runtime', 'packets.csl', 'packets.csl'),
        ('runtime', 'activation_stream.csl', 'activation_stream.csl'),
        ('runtime', 'token_sequence.csl', 'sequence.csl')]]
    for path, name in files:
        shutil.copy2(path, out / name)
    for name in data_files:
        if (out / name).exists():
            raise ValueError(f'Static input collides with program source: {name}')
        shutil.copy2(static_inputs / name, out / name)
    if static_inputs is not None:
        shutil.copy2(static_inputs / 'manifest.json', out / 'static-input-manifest.json')
        shutil.copy2(static_inputs / 'config.json', out / 'static-input-config.json')
    (out / 'config.json').write_text(json.dumps(dict(
        **limits, width=193, height=210, matrix_tiles=34552, kv_tiles=3168,
        controllers=25, layers=24, vocabulary=151936,
        initialization='selected-original-declarations' if data else 'zero',
        weight_storage=weight_storage,
        static_input_binding=binding_report,
        identity_initialization='Opt-in device application-coordinate derivation in initialize; unexecuted',
        scope=('Complete original-sized193x210 model CSL graph. '+
              ('Selected original matrix declarations, original norms/frequencies on all25controllers '
               '(frequencies on24decodercontrollers) and selected original bias roots; remaining matrices/biases zero. '
               if data else 'Mutable zero weights and unloaded norms. ')+
              'Unexecuted coordinate identity initialization and optional byte ABI. Compile-only '
              'role/route/resource qualification; no complete initialized model, SdkRuntime or neural result.')), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
        files={p.name: digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    from executor import verify
    verify(root)
    os.chdir(root)
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=200,212',
               '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1',
               '--max-parallelism=1', '--dump-dsr-alloc-graph']
    (root / 'compile-command.json').write_text(json.dumps(command)+'\n')
    subprocess.run(command, check=True)
    images = list((root / 'out/bin').glob('*.elf'))
    assert images
    (root / 'results.json').write_text(json.dumps(dict(
        success=True, compile_only=True, elf_count=len(images),
        scope=json.loads((root / 'config.json').read_text())['scope']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--weight-storage', choices=['u32', 'u8'], default='u32')
    parser.add_argument('--static-inputs', type=Path)
    parser.add_argument('--rss-limit-gib', type=int, choices=[8, 12], default=8,
                        help='Freeze the compile process-tree RSS guard at preparation')
    args = parser.parse_args()
    if args.prepare:
        prepare(args.source_root.resolve(), args.project_root.resolve(), args.weight_storage,
                args.static_inputs.resolve() if args.static_inputs else None, args.rss_limit_gib)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute, verify
        root = args.execute.resolve()
        verify(root)
        config = json.loads((root / 'config.json').read_text())
        kib = config['rss_limit_kib']
        assert type(kib) is int and kib % (1024*1024) == 0
        limits = resource_limits(kib // (1024*1024))
        assert config['timeout_seconds'] == limits['timeout_seconds']
        execute(root, limits['timeout_seconds'], rss_limit_kib=limits['rss_limit_kib'])

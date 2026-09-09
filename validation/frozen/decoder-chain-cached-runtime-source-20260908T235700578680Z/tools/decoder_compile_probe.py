"""Compile-only qualification of mutable resident-decoder streamed code paths.

No SdkRuntime is constructed, weights are not initialized, and no numerical or
delivery claim follows from this probe. The full 64x20 local routes are emitted.
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


def prepare(source, project):
    sys.path.insert(0, str(source / 'tools'))
    import resident_decoder_probe as decoder
    out = project / 'evidence' / ('decoder-stream-compile-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    standalone = decoder.make_layout(decoder.geometry(), initialization='sdk')
    exports = [line for line in standalone.splitlines() if line.startswith('@export_name')]
    fragment = decoder.make_layout(decoder.geometry(), fragment=True, next_destination=(63, 0), initialization='sdk')
    (out / 'layout.csl').write_text('const memcpy=@import_module("<memcpy/get_params>",.{.width=64,.height=20});\nlayout{@set_rectangle(64,20);\n' + fragment + '\n'.join(exports) + '\n}\n')
    files = [(Path(__file__), 'driver.py'), (source / 'tools/sdk_probe.py', 'executor.py'),
             (source / 'tools/resident_decoder_probe.py', 'builder.py'),
             (source / 'csl/programs/resident_decoder/pe.csl', 'pe.csl')]
    files += [(source / 'csl' / directory / name, target) for directory, name, target in [
        ('kernels', 'local_gemv_bf16_f32_colmajor.csl', 'kernel.csl'),
        ('kernels', 'vector_f32.csl', 'vector.csl'),
        ('kernels', 'qwen_position_f32.csl', 'rope.csl'),
        ('kernels', 'kv_block_f32.csl', 'kv.csl'),
        ('runtime', 'line_allreduce_f32.csl', 'line.csl'),
        ('runtime', 'filtered_input.csl', 'input.csl'),
        ('runtime', 'packets.csl', 'packets.csl'),
        ('runtime', 'activation_stream.csl', 'activation_stream.csl')]]
    for path, name in files:
        shutil.copy2(path, out / name)
    (out / 'config.json').write_text(json.dumps(dict(scope='Compile-only 64x20 resident decoder, mutable weights/auxiliaries/runtime IDs and streamed controller input/output. No initialized weights, execution, destination consumption or numerical acceptance.'), indent=2) + '\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d',
        files={p.name: digest(p) for p in out.iterdir() if p.is_file()}), indent=2) + '\n')
    print(out, flush=True)


def worker(root):
    from executor import verify
    verify(root); os.chdir(root)
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=71,22', '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1', '--max-parallelism=1', '--dump-dsr-alloc-graph']
    (root / 'compile-command.json').write_text(json.dumps(command) + '\n')
    subprocess.run(command, check=True)
    elfs = list((root / 'out/bin').glob('*.elf'))
    assert elfs
    (root / 'results.json').write_text(json.dumps(dict(success=True, compile_only=True, elf_count=len(elfs),
        scope=json.loads((root / 'config.json').read_text())['scope']), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--prepare', action='store_true')
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.prepare:
        prepare(args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 240, rss_limit_kib=4*1024*1024)

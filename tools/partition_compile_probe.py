"""Two original tiles: bounded intact-ELF composition versus direct compilation.

Compile/static audit only. Actual SDK execution is a separate frozen step.
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


def prepare(parent, template, source, project):
    import numpy as np
    sys.path.insert(0, str(source / 'tools'))
    from partition_encoder import encode_initializer
    from partition_executor import verify
    verify(parent)
    verify(template)
    out = project / 'evidence' / ('partition-compile-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    with (parent / 'weights.u32.bin').open('rb') as stream:
        payload = stream.read(2*28672)
    assert len(payload) == 2*28672
    words = np.frombuffer(payload, dtype='<u4').reshape(2, 7168).copy()
    head = np.load(parent / 'inputs.npz')['inputs'][0, :224].reshape(2, 112).copy()
    inputs = np.stack([head, -head, np.zeros_like(head), head])
    matrices = (words.view(np.uint16).reshape(2, 112, 128).astype(np.uint32) << 16).view(np.float32)
    expected = np.einsum('cpi,pio->cpo', inputs.astype(np.float64), matrices.astype(np.float64))
    np.savez(out / 'inputs.npz', weights=words, inputs=inputs, expected=expected)
    pe = (template / 'pe.csl').read_text()
    before = 'const initial=@import_module("weights.csl");'
    assert pe.count(before) == 1
    pe = 'param weight_file;\n' + pe.replace(before, 'const initial=@import_module(weight_file);')
    for variant in ['batch0', 'batch1', 'baseline']:
        directory = out / variant
        directory.mkdir()
        (directory / 'pe.csl').write_text(pe)
        shutil.copy2(template / 'kernel.csl', directory / 'kernel.csl')
        for tile in range(2):
            (directory / f'weights_{tile}.csl').write_text(encode_initializer(
                payload[tile*28672:(tile+1)*28672], 'bytes'))
        (directory / 'zero.csl').write_text('const values=@zeros([28672]u8);\n')
        layout = 'const memcpy=@import_module("<memcpy/get_params>",.{.width=2,.height=1});layout{@set_rectangle(2,1);'
        for tile in range(2):
            name = f'weights_{tile}.csl' if variant in [f'batch{tile}', 'baseline'] else 'zero.csl'
            layout += f'@set_tile_code({tile},0,"pe.csl",.{{.memcpy_params=memcpy.get_params({tile}),.weight_file="{name}"}});'
        for name, kind in [('weights', 'u8'), ('input', 'f32'), ('output', 'f32'), ('progress', 'u32'), ('timing', 'u16')]:
            layout += f'@export_name("{name}",[*]{kind},false);'
        (directory / 'layout.csl').write_text(layout+'@export_name("compute",fn()void);}\n')
    for path, name in [(Path(__file__), 'driver.py'), (source / 'tools/partition_executor.py', 'executor.py'),
                       (source / 'tools/elf_partition.py', 'elf_partition.py'),
                       (source / 'tools/compare_elf_images.py', 'compare_elf_images.py')]:
        shutil.copy2(path, out / name)
    (out / 'config.json').write_text(json.dumps(dict(
        parent=parent.name, parent_manifest_sha256=digest(parent / 'manifest.json'),
        template=template.name, template_manifest_sha256=digest(template / 'manifest.json'),
        cases=['canonical-shards', 'negative', 'zero', 'repeat'],
        gate=dict(relative_l2=2e-6, relative_peak=3e-6),
        scope='Two original tied matrix tiles, direct compilation and two full-layout '
              'partitions. Select/copy complete unmodified ELF files; require identical '
              'SDK loader segments/coordinates/sections/RPC/I/O to direct baseline. '
              'Compile-only experimental artifact composition, no SdkRuntime or model acceptance.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(
        sdk_sha256=json.loads((template / 'manifest.json').read_text())['sdk_sha256'],
        files={str(p.relative_to(out)): digest(p) for p in out.rglob('*') if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    from executor import verify
    from elf_partition import compose, image, io_signature, rpc
    verify(root)
    os.chdir(root)
    commands = []
    for variant in ['batch0', 'batch1', 'baseline']:
        command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=9,3',
                   '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1',
                   '--max-parallelism=1', '--dump-dsr-alloc-graph']
        commands.append(dict(directory=variant, command=command))
        (root / 'compile-commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        subprocess.run(command, cwd=root / variant, check=True)
    report = compose([(root / 'batch0/out', {(4, 1)}),
                      (root / 'batch1/out', {(5, 1)})], root / 'out', {(4, 1), (5, 1)})
    (root / 'composition.json').write_text(json.dumps(report, indent=2)+'\n')

    def images(directory):
        result = {}
        for path in (directory / 'bin').glob('*.elf'):
            value = image(path)
            for point in value['coordinates']:
                point = tuple(point)
                assert point not in result
                result[point] = value
        return result

    assert images(root / 'out') == images(root / 'baseline/out')
    assert io_signature(root / 'out') == io_signature(root / 'baseline/out')
    assert rpc(root / 'out/bin/out_rpc.json') == rpc(root / 'baseline/out/bin/out_rpc.json')
    (root / 'results.json').write_text(json.dumps(dict(
        success=True, compile_only=True, full_direct_baseline_loader_image_equal=True,
        composition_sha256=digest(root / 'composition.json'),
        scope=json.loads((root / 'config.json').read_text())['scope']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-fixture', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--template', type=Path)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.from_fixture:
        if not args.template:
            parser.error('--template is required for preparation')
        prepare(args.from_fixture.resolve(), args.template.resolve(),
                args.source_root.resolve(), args.project_root.resolve())
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 120, rss_limit_kib=1024*1024)

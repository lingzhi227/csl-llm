"""Two-tile or eight-tile collective: bounded intact-ELF composition versus direct compilation.

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


def prepare(parent, template, source, project, collective=False):
    import numpy as np
    sys.path.insert(0, str(source / 'tools'))
    from partition_encoder import encode_initializer
    from partition_executor import verify
    verify(parent)
    verify(template)
    out = project / 'evidence' / ('partition-compile-' + datetime.datetime.now(
        datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    width = 8 if collective else 2
    partitions = [list(range(i, i+2)) for i in range(0, 8, 2)] if collective else [[0], [1]]
    with (parent / 'weights.u32.bin').open('rb') as stream:
        payload = stream.read(width*28672)
    assert len(payload) == width*28672
    words = np.frombuffer(payload, dtype='<u4').reshape(width, 7168).copy()
    head = np.load(parent / 'inputs.npz')['inputs'][0, :width*112].reshape(width, 112).copy()
    inputs = np.stack([head, -head, np.zeros_like(head), head])
    matrices = (words.view(np.uint16).reshape(width, 112, 128).astype(np.uint32) << 16).view(np.float32)
    expected = np.einsum('cpi,pio->cpo', inputs.astype(np.float64), matrices.astype(np.float64))
    if collective:
        expected = np.repeat(expected.sum(axis=1)[:, None, :], width, axis=1)
    np.savez(out / 'inputs.npz', weights=words, inputs=inputs, expected=expected)
    pe = (template / 'pe.csl').read_text()
    before = 'const initial=@import_module("weights.csl");'
    assert pe.count(before) == 1
    pe = 'param weight_file;\n' + pe.replace(before, 'const initial=@import_module(weight_file);')
    if collective:
        pe = (source / 'csl/programs/partition_collective/pe.csl').read_text()
    for variant in [f'batch{i}' for i in range(len(partitions))]+['baseline']:
        directory = out / variant
        directory.mkdir()
        (directory / 'pe.csl').write_text(pe)
        shutil.copy2(template / 'kernel.csl', directory / 'kernel.csl')
        if collective:
            shutil.copy2(source / 'csl/programs/partition_collective/line.csl', directory / 'line.csl')
        for tile in range(width):
            (directory / f'weights_{tile}.csl').write_text(encode_initializer(
                payload[tile*28672:(tile+1)*28672], 'bytes'))
        (directory / 'zero.csl').write_text('const values=@zeros([28672]u8);\n')
        layout = f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height=1}});layout{{@set_rectangle({width},1);'
        for tile in range(width):
            selected = variant == 'baseline' or tile in partitions[int(variant[5:])]
            name = f'weights_{tile}.csl' if selected else 'zero.csl'
            extra = f',.ordinal={tile}' if collective else ''
            layout += f'@set_tile_code({tile},0,"pe.csl",.{{.memcpy_params=memcpy.get_params({tile}),.weight_file="{name}"{extra}}});'
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
        width=width, partitions=partitions, collective=collective,
        scope=f'{width} original tied matrix tiles, collective={collective}, direct compilation and full-layout '
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
    config = json.loads((root / 'config.json').read_text())
    width = config.get('width', 2)
    partitions = config.get('partitions', [[0], [1]])
    commands = []
    for variant in [f'batch{i}' for i in range(len(partitions))]+['baseline']:
        command = ['cslc', 'layout.csl', '--arch=wse3', f'--fabric-dims={width+7},3',
                   '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1',
                   '--max-parallelism=1', '--dump-dsr-alloc-graph']
        commands.append(dict(directory=variant, command=command))
        (root / 'compile-commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        subprocess.run(command, cwd=root / variant, check=True)
    report = compose([(root / f'batch{i}/out', {(x+4, 1) for x in points})
                      for i, points in enumerate(partitions)], root / 'out',
                     {(x+4, 1) for x in range(width)})
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
    parser.add_argument('--collective', action='store_true')
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.from_fixture:
        if not args.template:
            parser.error('--template is required for preparation')
        prepare(args.from_fixture.resolve(), args.template.resolve(),
                args.source_root.resolve(), args.project_root.resolve(), args.collective)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 120, rss_limit_kib=1024*1024)

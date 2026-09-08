"""Bounded compact-initializer scaling on the unchanged full vocabulary program.

Only selected original tiles are statically initialized; every other matrix
remains mutable zero storage. This tool never constructs SdkRuntime.
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(parent, project, source, count, representation='packed', debug_info=True, start_tile=0):
    if not 1 <= count <= 1024:
        raise ValueError('This bounded scale experiment permits at most1024 static tiles')
    if not 0 <= start_tile <= 9496-count:
        raise ValueError('Static tile interval must lie within the original9496 matrix tiles')
    sys.path.insert(0, str(source / 'tools'))
    from compact_weights_probe import encode_initializer
    manifest = json.loads((parent / 'manifest.json').read_text())
    config = json.loads((parent / 'config.json').read_text())
    assert config['matrix_tiles'] == 9496 and config['initialization'] == 'sdk'
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(parent / name) != sha:
            raise ValueError(f'Changed original fixture {name}')
    out = project / 'evidence' / ('compact-vocabulary-compile-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    for name in manifest['files']:
        if name.endswith('.csl') and name != 'layout.csl':
            shutil.copy2(parent / name, out / name)
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/sdk_probe.py', out / 'executor.py')
    shutil.copy2(source / 'tools/compact_weights_probe.py', out / 'encoder.py')
    with (parent / 'weights.u32.bin').open('rb') as stream:
        stream.seek(start_tile*28672)
        payload = stream.read(count*28672)
    assert len(payload) == count*28672
    (out / 'expected.bin').write_bytes(payload)
    for index in range(count):
        x, y = (start_tile+index) % 192, (start_tile+index) // 192
        (out / f'weights_{x}_{y}.csl').write_text(encode_initializer(payload[index*28672:(index+1)*28672], representation))
    if representation == 'bytes':
        pe = (out / 'pe.csl').read_text()
        before = 'var weights=if(matrix and static_weights) initial.values else @zeros([if(matrix) 7168 else 1]u32);'
        assert pe.count(before) == 1
        pe = pe.replace(before, 'var weights=if(matrix and static_weights) initial.values else @zeros([if(matrix) 28672 else 4]u8);')
        assert pe.count('const weight_ptr:[*]u32=&weights;') == 1
        pe = pe.replace('const weight_ptr:[*]u32=&weights;', 'const weight_ptr:[*]u8=&weights;')
        (out / 'pe.csl').write_text(pe)
    lines = []
    changed = 0
    for line in (parent / 'layout.csl').read_text().splitlines():
        if line.startswith('@set_tile_code'):
            x, y = map(int, re.match(r'@set_tile_code\((\d+),(\d+),', line).groups())
            if x < 192 and start_tile <= x+y*192 < start_tile+count:
                assert '.matrix=true' in line and '.static_weights=false' in line
                line = line.replace('.weight_file="<empty>"', f'.weight_file="weights_{x}_{y}.csl"')
                line = line.replace('.static_weights=false', '.static_weights=true')
                changed += 1
        lines.append(line)
    assert changed == count
    if representation == 'bytes':
        lines = [line.replace('@export_name("weights",[*]u32,false);', '@export_name("weights",[*]u8,false);') for line in lines]
    (out / 'layout.csl').write_text('\n'.join(lines)+'\n')
    (out / 'config.json').write_text(json.dumps(dict(parent=parent.name,
        parent_manifest_sha256=digest(parent / 'manifest.json'), source_packed_sha256=digest(parent / 'weights.u32.bin'),
        static_tile_indices=list(range(start_tile, start_tile+count)), matrix_tiles=9496, representation=representation,
        debug_info=debug_info,
        timeout_seconds=240 if count <= 256 else 900,
        rss_limit_kib=(4 if count <= 256 else 12)*1024*1024,
        scope='Compile-only full193x50 vocabulary program, declared physical row-major tile interval statically initialized from original compact byte strings; all remaining matrix weights zero. Exact allocated ELF coordinate audit, no runtime or full initialization/model numerical acceptance.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    from executor import verify
    from cerebras.elf.cself import ELFMemory
    from elftools.elf.elffile import ELFFile
    verify(root);os.chdir(root)
    config = json.loads((root / 'config.json').read_text())
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=200,52', '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1', '--max-parallelism=1', '--dump-dsr-alloc-graph']
    if not config.get('debug_info', True):
        command.append('--g0')
    (root / 'compile-command.json').write_text(json.dumps(command)+'\n')
    subprocess.run(command, check=True)
    config = json.loads((root / 'config.json').read_text())
    count = len(config['static_tile_indices'])
    indices = config['static_tile_indices']
    assert indices == list(range(indices[0], indices[0]+count))
    assert 0 <= indices[0] and indices[-1] < 9496
    selected_offsets = {index: offset for offset, index in enumerate(indices)}
    expected = (root / 'expected.bin').read_bytes()
    assert len(expected) == count*28672
    seen, images = set(), []
    for path in sorted((root / 'out/bin').glob('*.elf')):
        coordinates = [(x-4, y-1) for x, y in ELFMemory(str(path)).iter_coordinates()]
        selected = [(x, y) for x, y in coordinates if 0 <= x < 192 and 0 <= y < 50 and x+192*y < 9496]
        if not selected:
            continue
        with path.open('rb') as stream:
            elf = ELFFile(stream)
            symbols = elf.get_section_by_name('.symtab').get_symbol_by_name('weights')
            assert symbols and len(symbols) == 1
            symbol = symbols[0];assert symbol['st_size'] == 28672
            assert symbol['st_value'] % 4 == 0, ('weight alignment', path.name)
            section = elf.get_section(symbol['st_shndx'])
            offset = symbol['st_value']-section['sh_addr']
            assert 0 <= offset and offset+28672 <= section['sh_size']
            raw = section.data()[offset:offset+28672]
        for x, y in selected:
            index = x+192*y
            assert index not in seen
            offset = selected_offsets.get(index)
            wanted = expected[offset*28672:(offset+1)*28672] if offset is not None else bytes(28672)
            assert raw == wanted, (path.name, x, y)
            seen.add(index)
        images.append(dict(file=path.name, sha256=digest(path), coordinates=selected,
            weights_sha256=hashlib.sha256(raw).hexdigest()))
    assert seen == set(range(9496))
    (root / 'results.json').write_text(json.dumps(dict(success=True, compile_only=True,
        static_original_tiles=count, zero_matrix_tiles=9496-count, all_matrix_coordinates_exact=True,
        elf_count=len(list((root / 'out/bin').glob('*.elf'))), images=images, scope=config['scope']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-fixture', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--tiles', type=int, default=256)
    parser.add_argument('--start-tile', type=int, default=0)
    parser.add_argument('--representation', choices=['packed', 'bytes'], default='packed')
    parser.add_argument('--no-debug-info', action='store_true',
                        help='Use supported cslc --g0; keep all runtime audits and DSR graphs')
    args = parser.parse_args()
    if args.from_fixture:
        prepare(args.from_fixture.resolve(), args.project_root.resolve(), args.source_root.resolve(), args.tiles, args.representation, not args.no_debug_info, args.start_tile)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        config = json.loads((args.execute / 'config.json').read_text())
        execute(args.execute.resolve(), config.get('timeout_seconds', 240),
                rss_limit_kib=config.get('rss_limit_kib', 4*1024*1024))

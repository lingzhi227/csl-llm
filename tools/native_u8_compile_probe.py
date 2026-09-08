"""Compile-only original tile string initializer; exact allocated ELF audit."""
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


def encode_initializer(payload, representation='packed'):
    if len(payload) != 28672:
        raise ValueError('Original tile must contain28672bytes')
    encoded = ''.join(f'\\x{byte:02x}' for byte in payload)
    if representation == 'bytes':
        return 'const values=@get_array("'+encoded+'");\n'
    if representation != 'packed':
        raise ValueError('Unknown compact initializer representation')
    return 'const bytes=@get_array("'+encoded+'");\n'+'''fn unpack() [7168]u32 {
 var words=@zeros([7168]u32);
 for(@range(u16,7168))|i|{
  const j=i*4;
  words[i]=@as(u32,bytes[j])|(@as(u32,bytes[j+1])<<8)|(@as(u32,bytes[j+2])<<16)|(@as(u32,bytes[j+3])<<24);
 }
 return words;
}
const values=comptime unpack();
'''


def prepare(parent, project, source, representation='packed'):
    manifest = json.loads((parent / 'manifest.json').read_text())
    path = parent / 'weights.u32.bin'
    if digest(path) != manifest['files']['weights.u32.bin']:
        raise ValueError('Parent original packed weights changed')
    with path.open('rb') as stream:
        payload = stream.read(28672)
    assert len(payload) == 28672
    out = project / 'evidence' / ('compact-weights-compile-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    (out / 'expected.bin').write_bytes(payload)
    (out / 'weights.csl').write_text(encode_initializer(payload, representation))
    kind = 'u8' if representation == 'bytes' else 'u32'
    (out / 'pe.csl').write_text('param memcpy_params;const sys=@import_module("<memcpy/memcpy>",memcpy_params);const initial=@import_module("weights.csl");var weights=initial.values;const wp:[*]'+kind+'=&weights;fn noop()void{sys.unblock_cmd_stream();}comptime{@export_symbol(wp,"weights");@export_symbol(noop);}\n')
    (out / 'layout.csl').write_text('const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});@export_name("weights",[*]'+kind+',false);@export_name("noop",fn()void);}\n')
    shutil.copy2(Path(__file__), out / 'driver.py')
    shutil.copy2(source / 'tools/native_u8_executor.py', out / 'executor.py')
    (out / 'config.json').write_text(json.dumps(dict(parent=parent.name, parent_manifest_sha256=digest(parent / 'manifest.json'),
        source_packed_sha256=digest(path), tile_index=0, byte_offset=0, bytes=28672, representation=representation,
        scope='One actual original128x112 BF16 tile compiled from a byte-string constant and compared with allocated ELF contents. No SdkRuntime, numerical computation or large-scale compiler-memory qualification.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


def worker(root):
    from executor import verify
    from elftools.elf.elffile import ELFFile
    verify(root);os.chdir(root)
    command = ['cslc', 'layout.csl', '--arch=wse3', '--fabric-dims=8,3', '--fabric-offsets=4,1', '-o=out', '--memcpy', '--channels=1', '--max-parallelism=1', '--dump-dsr-alloc-graph']
    (root / 'compile-command.json').write_text(json.dumps(command)+'\n')
    subprocess.run(command, check=True)
    elfs = list((root / 'out/bin').glob('*.elf'))
    assert len(elfs) == 1
    with elfs[0].open('rb') as stream:
        elf = ELFFile(stream)
        symbols = [symbol for symbol in elf.get_section_by_name('.symtab').iter_symbols()
                   if symbol.name == 'weights']
        assert len(symbols) == 1
        symbol = symbols[0];assert symbol['st_size'] == 28672
        section = elf.get_section(symbol['st_shndx'])
        start = symbol['st_value']-section['sh_addr']
        actual = section.data()[start:start+symbol['st_size']]
        assert symbol['st_value'] % 4 == 0
    assert actual == (root / 'expected.bin').read_bytes()
    (root / 'results.json').write_text(json.dumps(dict(success=True, compile_only=True,
        initializer_exact=True, elf_sha256=digest(elfs[0]), actual_bytes_sha256=hashlib.sha256(actual).hexdigest(),
        scope=json.loads((root / 'config.json').read_text())['scope']), indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-fixture', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    parser.add_argument('--representation', choices=['packed', 'bytes'], default='packed')
    args = parser.parse_args()
    if args.from_fixture:
        prepare(args.from_fixture.resolve(), args.project_root.resolve(), args.source_root.resolve(), args.representation)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 60, rss_limit_kib=1024*1024)

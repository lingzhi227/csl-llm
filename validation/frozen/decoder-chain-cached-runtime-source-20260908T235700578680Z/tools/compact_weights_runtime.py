"""Actual SDK u8-export/raw-u32 readback and native GEMV qualification."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def prepare(parent, project, source):
    import numpy as np
    manifest = json.loads((parent / 'manifest.json').read_text())
    config = json.loads((parent / 'config.json').read_text())
    assert config['representation'] == 'bytes'
    for name, sha in manifest['files'].items():
        if Path(name).name != name or digest(parent / name) != sha:
            raise ValueError(f'Changed compact fixture {name}')
    original = project / 'evidence' / config['parent']
    original_manifest = json.loads((original / 'manifest.json').read_text())
    assert digest(original / 'manifest.json') == config['parent_manifest_sha256']
    assert digest(original / 'inputs.npz') == original_manifest['files']['inputs.npz']
    x = np.load(original / 'inputs.npz')['inputs'][0, :112].copy()
    words = np.frombuffer((parent / 'expected.bin').read_bytes(), dtype='<u4').copy()
    matrix = (words.view(np.uint16).reshape(112, 128).astype(np.uint32) << 16).view(np.float32).T
    inputs = np.stack([x, -x, np.zeros_like(x), x])
    expected = inputs.astype(np.float64) @ matrix.astype(np.float64).T
    out = project / 'evidence' / ('compact-weights-runtime-' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    out.mkdir()
    np.savez(out / 'inputs.npz', weights=words, inputs=inputs, expected=expected)
    copies = [(Path(__file__), 'driver.py'), (source / 'tools/sdk_probe.py', 'executor.py'),
              (source / 'tools/static_weights_probe.py', 'runtime_worker.py'),
              (source / 'csl/kernels/local_gemv_bf16_f32_colmajor.csl', 'kernel.csl'),
              (parent / 'weights.csl', 'weights.csl')]
    for path, name in copies:
        shutil.copy2(path, out / name)
    (out / 'pe.csl').write_text('''param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const clock=@import_module("<time>");
const kernel=@import_module("kernel.csl",.{.rows=128,.columns=112});
const initial=@import_module("weights.csl");var weights=initial.values;
var input=@zeros([112]f32);var output=@zeros([128]f32);var progress=@zeros([1]u32);
var timing=@zeros([6]u16);var started=@zeros([3]u16);var ended=@zeros([3]u16);
fn compute() void {
 clock.enable_tsc();clock.get_timestamp(&started);
 kernel.compute(@ptrcast([*]u16,&weights),&input,&output);
 clock.get_timestamp(&ended);clock.disable_tsc();
 for(@range(u16,3))|i|{timing[i]=started[i];timing[i+3]=ended[i];}
 progress[0]+=1;sys.unblock_cmd_stream();
}
const wp:[*]u8=&weights;const ip:[*]f32=&input;const op:[*]f32=&output;
const pp:[*]u32=&progress;const tp:[*]u16=&timing;
comptime{@export_symbol(wp,"weights");@export_symbol(ip,"input");@export_symbol(op,"output");@export_symbol(pp,"progress");@export_symbol(tp,"timing");@export_symbol(compute);}
''')
    exports = [('weights', 'u8'), ('input', 'f32'), ('output', 'f32'), ('progress', 'u32'), ('timing', 'u16')]
    (out / 'layout.csl').write_text('const memcpy=@import_module("<memcpy/get_params>",.{.width=1,.height=1});layout{@set_rectangle(1,1);@set_tile_code(0,0,"pe.csl",.{.memcpy_params=memcpy.get_params(0)});' + ''.join(f'@export_name("{name}",[*]{kind},false);' for name, kind in exports) + '@export_name("compute",fn()void);}\n')
    (out / 'config.json').write_text(json.dumps(dict(source_bundle=parent.name,
        source_manifest_sha256=digest(parent / 'manifest.json'), original_inputs_sha256=digest(original / 'inputs.npz'),
        cases=['canonical-shard', 'negative', 'zero', 'repeat'], simulator_threads=1,
        gate=dict(relative_l2=2e-6, relative_peak=3e-6),
        scope='One original tied-vocabulary128x112 BF16 tile stored asu8 CSL byte-string initializer. Actual SDK get_id/MEMCPY_32BIT fullrawreadback before/after4native GEMVs using runtimeu16pointerview. Already-normalized original reference shard input plusnegative/zero/repeat; not fullvocab or model inference.'), indent=2)+'\n')
    (out / 'manifest.json').write_text(json.dumps(dict(sdk_sha256=manifest['sdk_sha256'],
        files={p.name:digest(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    print(out, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--from-fixture', type=Path)
    action.add_argument('--worker', type=Path)
    action.add_argument('--execute', type=Path)
    parser.add_argument('--project-root', type=Path, default=ROOT)
    parser.add_argument('--source-root', type=Path, default=ROOT)
    args = parser.parse_args()
    if args.from_fixture:
        prepare(args.from_fixture.resolve(), args.project_root.resolve(), args.source_root.resolve())
    elif args.worker:
        sys.path.insert(0, str(args.worker.resolve()))
        from runtime_worker import worker
        worker(args.worker.resolve())
    else:
        sys.path.insert(0, str(args.execute.resolve()))
        from executor import execute
        execute(args.execute.resolve(), 120, rss_limit_kib=1024*1024)

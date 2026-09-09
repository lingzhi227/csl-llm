"""Freeze target-local-size GEMV probe; replicated layouts measure simulator cost."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
import numpy as np
from safetensors.torch import load_file
ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--rows',type=int,default=32);p.add_argument('--columns',type=int,default=224);p.add_argument('--width',type=int,default=1);p.add_argument('--height',type=int,default=1);p.add_argument('--colmajor',action='store_true');p.add_argument('--varied-inputs',action='store_true');p.add_argument('--sampled',action='store_true');p.add_argument('--storage',choices=['f32','bf16'],default='f32');p.add_argument('--real-trace',type=Path);a=p.parse_args();assert 1<=a.width<=256 and 1<=a.height<=256;assert 1<=a.rows<=256 and 1<=a.columns<=896;assert not a.colmajor or a.storage=="bf16"
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/'evidence'/('gemv-'+a.storage+'-'+str(a.width)+'x'+str(a.height)+'-'+stamp);out.mkdir()
    shutil.copy2(ROOT/('csl/kernels/local_gemv_'+(('bf16_f32_colmajor' if a.colmajor else 'bf16_f32') if a.storage=='bf16' else 'f32')+'.csl'),out/'local_gemv_f32.csl');shutil.copy2(ROOT/'tools/sdk_probe.py',out/'driver.py');shutil.copy2(__file__,out/'prepare.py')
    real=a.real_trace is not None;assert not real or a.width*a.height==1
    if real:
        model=ROOT/'models/qwen2.5-0.5b-7ae5576/model.safetensors';weights=load_file(model)['model.layers.0.mlp.up_proj.weight'][:a.rows,:a.columns].float().numpy()
        inputs=np.load(a.real_trace)['model.layers.0.post_attention_layernorm'][0,-1,:a.columns]
    else:weights=np.full((a.rows,a.columns),1/64,np.float32);inputs=np.full(a.columns,1/8,np.float32)
    expected=weights.astype(np.float64)@inputs.astype(np.float64)
    input_batches=[inputs.copy() for _ in range(3)]
    if a.varied_inputs:
        assert real
        input_batches=[inputs.copy()]
        for index in (0,1,a.columns//2,a.columns-2,a.columns-1):
            v=np.zeros(a.columns,np.float32);v[index]=1.0;input_batches.append(v)
        input_batches.extend([np.zeros(a.columns,np.float32),-inputs.copy(),inputs.copy()])
    np.savez(out/'inputs.npz',input_batches=np.stack(input_batches),expected_batches=np.stack([weights.astype(np.float64)@v.astype(np.float64) for v in input_batches]),weights=weights,weights_u16=(weights.view(np.uint32)>>16).astype(np.uint16),input=inputs,expected=expected)
    (out/'layout.csl').write_text(f'''const memcpy=@import_module("<memcpy/get_params>",.{{.width={a.width},.height={a.height}}});
layout{{
 @set_rectangle({a.width},{a.height});
 for(@range(i16,{a.width}))|x|{{for(@range(i16,{a.height}))|y|{{@set_tile_code(x,y,"pe.csl",.{{.memcpy_params=memcpy.get_params(x)}});}}}}
 @export_name("weights",[*]f32,false);@export_name("input",[*]f32,false);@export_name("output",[*]f32,false);@export_name("progress",[*]u32,false);@export_name("timing",[*]u16,false);@export_name("compute",fn()void);
}}
''')
    (out/'pe.csl').write_text('''param memcpy_params;
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const clock=@import_module("<time>");
const gemv=@import_module("local_gemv_f32.csl",.{.rows=32,.columns=224});
var weights=@constants([7168]f32,0.015625);var input=@constants([224]f32,0.125);var output=@zeros([32]f32);
var progress=@zeros([1]u32);var timing=@zeros([6]u16);
fn compute() void {
 var start=@zeros([3]u16);var end=@zeros([3]u16);
 clock.enable_tsc();clock.get_timestamp(&start);gemv.compute(&weights,&input,&output);clock.get_timestamp(&end);clock.disable_tsc();
 for(@range(u16,3))|i|{timing[i]=start[i];timing[i+3]=end[i];}
 progress[0]+=1;sys.unblock_cmd_stream();
}
const weights_ptr:[*]f32=&weights;const input_ptr:[*]f32=&input;const output_ptr:[*]f32=&output;const progress_ptr:[*]u32=&progress;const timing_ptr:[*]u16=&timing;
comptime{@export_symbol(weights_ptr,"weights");@export_symbol(input_ptr,"input");@export_symbol(output_ptr,"output");@export_symbol(progress_ptr,"progress");@export_symbol(timing_ptr,"timing");@export_symbol(compute);}
''')
    pe=(out/'pe.csl').read_text().replace('.rows=32,.columns=224',f'.rows={a.rows},.columns={a.columns}').replace('[7168]f32',f'[{a.rows*a.columns}]f32').replace('[224]f32',f'[{a.columns}]f32').replace('[32]f32',f'[{a.rows}]f32');(out/'pe.csl').write_text(pe)
    if a.storage=='bf16':
        pe=(out/'pe.csl').read_text().replace(f'var weights=@constants([{a.rows*a.columns}]f32,0.015625);',f'var weights=@constants([{a.rows*a.columns}]u16,0x3c80);').replace('const weights_ptr:[*]f32','const weights_ptr:[*]u16');(out/'pe.csl').write_text(pe)
        layout=(out/'layout.csl').read_text().replace('@export_name("weights",[*]f32','@export_name("weights",[*]u16');(out/'layout.csl').write_text(layout)
    (out/'config.json').write_text(json.dumps(dict(colmajor=a.colmajor,sampled=a.sampled,storage=a.storage,width=a.width,height=a.height,real_weights=real,calls=len(input_batches),threads=8,rows=a.rows,columns=a.columns,source_binding=dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256(a.real_trace.read_bytes()).hexdigest(),tensor='model.layers.0.mlp.up_proj.weight',weight_slice=[[0,a.rows],[0,a.columns]],activation='model.layers.0.post_attention_layernorm',activation_position='last prompt token',activation_slice=[0,a.columns]) if real else None,scope='Real Qwen local up-projection slice' if real else 'Replicated target-local-shape simulator cost probe; no full-model numerical claim',real_trace=str(a.real_trace) if real else None,gate=dict(relative_l2=2e-6,relative_peak=3e-6)),indent=2)+'\n')
    files={x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in out.iterdir() if x.is_file()}
    (out/'manifest.json').write_text(json.dumps(dict(files=files,sdk_sha256='fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d'),indent=2)+'\n');print(out,flush=True)
if __name__=='__main__':main()

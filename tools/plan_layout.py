import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from csl_llm.layout import plan
p=argparse.ArgumentParser();p.add_argument('inventory',type=Path);p.add_argument('output',type=Path);a=p.parse_args();assert not a.output.exists()
r=plan(json.loads(a.inventory.read_text()));r['inventory_sha256']=hashlib.sha256(a.inventory.read_bytes()).hexdigest();r['planner_sha256']=hashlib.sha256((ROOT/'src/csl_llm/layout.py').read_bytes()).hexdigest();a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['role_counts'],{k:sum(v.values()) for k,v in r['estimated_memory_per_pe'].items()})

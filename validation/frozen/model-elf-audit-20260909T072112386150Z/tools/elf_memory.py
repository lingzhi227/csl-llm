"""Classify every allocatable ELF section; never silently discard overflow."""
import argparse,hashlib,json
from pathlib import Path

# Pinned WSE3 compiler-emitted configuration windows observed in actual ELFs.
# These are retained in the report, not counted as SRAM. Unknown names/locations
# require explicit review; this is not a general ISA memory-map specification.
CONFIG_WINDOWS={
 '.entry_ival':(0xfc3c,2),'.fpcw':(0xfc18,2),'.fscale':(0xfc1a,2),
 '.blocked_ival':(0xfc08,4),'.active_ival':(0xfc0c,4),'.blocked_ut_ival':(0xfc04,4),
 '.fabric_routes':(0xf800,48),'.fabric_switches':(0xf900,192),
 '.ce_in_q':(0xf780,128),'.ce_out_q':(0xf700,128),'.prng_state':(0xfba0,16),'.filters':(0xf680,96),
 '.user_cfg_0':(0xfc10,2),'.user_cfg_1':(0xfc12,2),'.user_cfg_2':(0xfc24,2),'.user_cfg_3':(0xfc3e,2),
}
SRAM_NAMES={'.data.lo','.text','.rodata','.data','.bss','.task_table','.data.hi'}


def classify(section):
    name,start,size=section['name'],section['start'],section['size']
    assert type(start) is int and type(size) is int and start>=0 and size>=0
    if name in CONFIG_WINDOWS:
        address,capacity=CONFIG_WINDOWS[name]
        assert start==address and size<=capacity,('unexpected configuration window',section)
        return 'SDK configuration outside SRAM'
    assert name in SRAM_NAMES,('unrecognized allocatable section',section)
    assert start<=49152 and start+size<=49152,('SRAM overflow',section)
    return 'SRAM'


def inspect(path):
    from elftools.elf.elffile import ELFFile
    with path.open('rb') as stream:
        elf=ELFFile(stream);sections=[dict(name=s.name,start=int(s['sh_addr']),size=int(s['sh_size']),flags=int(s['sh_flags']),type=s['sh_type']) for s in elf.iter_sections() if s['sh_flags']&2]
    for s in sections:s['address_space']=classify(s)
    resident=[s for s in sections if s['address_space']=='SRAM'];high=max(s['start']+s['size'] for s in resident)
    return dict(elf=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),static_high_water_bytes=high,static_free_bytes=49152-high,sections=sections)


def main():
    p=argparse.ArgumentParser();p.add_argument('bundle',type=Path);p.add_argument('report',type=Path);a=p.parse_args();assert not a.report.exists();rows=[inspect(path) for path in sorted((a.bundle/'out/bin').glob('*.elf'))];assert rows
    report=dict(passed=True,classes=rows,max_static_high_water_bytes=max(r['static_high_water_bytes'] for r in rows),manifest_sha256=hashlib.sha256((a.bundle/'manifest.json').read_bytes()).hexdigest(),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='All allocatable sections listed and classified. Known SDK configuration windows explicitly retained; unrecognized sections or SRAM overflow fail. Static SRAM span is not a dynamic-stack, protocol or full-model fit proof.')
    a.report.write_text(json.dumps(report,indent=2)+'\n');print('ELF STATIC',len(rows),report['max_static_high_water_bytes'])
if __name__=='__main__':main()

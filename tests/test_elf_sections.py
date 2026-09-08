import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools'))
from elf_memory import classify
class MemorySections(unittest.TestCase):
    def test_explicit_address_spaces(self):
        self.assertEqual(classify(dict(name='.text',start=49150,size=2)),'SRAM')
        self.assertEqual(classify(dict(name='.fabric_routes',start=0xf800,size=48)),'SDK configuration outside SRAM')
        for record in [dict(name='.text',start=49152,size=1),dict(name='.bss',start=60000,size=1),dict(name='.mystery',start=0xf000,size=4),dict(name='.fabric_routes',start=49152,size=48),dict(name='.fabric_routes',start=0xf800,size=49)]:
            with self.assertRaises(AssertionError):classify(record)
if __name__=='__main__':unittest.main()

"""Contract integrity checks without loading or fabricating a runtime core."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
from contract_sdk_core import contract_symbols


class CoreContract(unittest.TestCase):
    def reports(self):
        name='out_0_0.elf';sha='a'*64
        contract=dict(passed=True,application_coordinates=1,images=[dict(
            elf=name,sha256=sha,coordinates=[[4,1]],
            required_symbols={'progress':dict(address=100,size=4)})])
        composition=dict(passed=True,elf_count=1,coordinate_count=1,
            artifact_sha256={'bin/'+name:sha},images=[dict(output=name,sha256=sha,
                allocated_image=dict(coordinates=[[4,1]],fabric_dimensions=[8,3],architecture_flags=2))])
        return contract,composition

    def test_reject_changed_elf_and_incomplete_or_duplicate_coordinates(self):
        contract,composition=self.reports()
        self.assertEqual(contract_symbols(contract,composition),
                         ({(4,1):{'progress':(100,4)}},((8,3),2)))
        changed=copy.deepcopy(contract);changed['images'][0]['sha256']='b'*64
        with self.assertRaises(ValueError):contract_symbols(changed,composition)
        changed=copy.deepcopy(contract);changed['images'][0]['coordinates']=[[5,1]]
        with self.assertRaises(ValueError):contract_symbols(changed,composition)
        changed=copy.deepcopy(contract);changed['images'].append(changed['images'][0])
        with self.assertRaises(ValueError):contract_symbols(changed,composition)
        changed=copy.deepcopy(contract);changed['application_coordinates']=2
        with self.assertRaises(ValueError):contract_symbols(changed,composition)

    def test_reject_out_of_sram_symbols_or_unbound_output_receipt(self):
        contract,composition=self.reports()
        contract['images'][0]['required_symbols']['progress']['address']=49150
        with self.assertRaises(ValueError):contract_symbols(contract,composition)
        contract,composition=self.reports();composition['artifact_sha256']={}
        with self.assertRaises(ValueError):contract_symbols(contract,composition)


if __name__=='__main__':unittest.main()

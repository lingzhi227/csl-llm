import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'tools'))
from bf16_transport import pack_bf16_pairs


class BF16Transport(unittest.TestCase):
    def test_all_patterns_and_byte_order(self):
        words=np.arange(65536,dtype=np.uint16)
        for raw in (words,words.astype('>u2'),words[::-1]):
            packed=pack_bf16_pairs(raw)
            np.testing.assert_array_equal(packed.view('<u2'),raw)
            self.assertEqual(packed.nbytes,raw.nbytes)
            self.assertEqual(int(packed[0]),int(raw[0])|(int(raw[1])<<16))

    def test_tile_boundaries_preserve_c_order(self):
        raw=np.arange(4*112*128,dtype=np.uint16).reshape(2,2,112,128)
        packed=pack_bf16_pairs(raw)
        np.testing.assert_array_equal(packed.view('<u2').reshape(raw.shape),raw)

    def test_reject_numeric_conversion_and_odd_payload(self):
        for raw in (np.ones(2,np.float32),np.ones(2,np.int16),np.ones(3,np.uint16)):
            with self.assertRaises(ValueError):pack_bf16_pairs(raw)


if __name__=='__main__':unittest.main()

import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from csl_llm.regions import Region,layout


class RegionGeometry(unittest.TestCase):
    def test_serpentine_boundaries(self):
        r=Region(3,2,11,6)
        points=[r.point(i) for i in range(66)]
        self.assertEqual(len(set(points)),66)
        self.assertEqual(points[10:13],[(13,2),(13,3),(12,3)])
        for left,right in zip(points,points[1:]):
            self.assertEqual(abs(left[0]-right[0])+abs(left[1]-right[1]),1)
        self.assertEqual(r.neighbors(10),('WEST','SOUTH'))
        self.assertEqual(r.neighbors(11),('NORTH','WEST'))

    def test_invalid_regions(self):
        for regions in [[Region(0,0,4,1),Region(3,0,4,1)],[Region(7,0,2,1)]]:
            with self.assertRaises(ValueError):layout(8,2,regions,[])


if __name__=='__main__':unittest.main()

import unittest
import sys
import numpy as np

try:
    sys.path.insert(0, "..")
    from src.yolo_utils import detect2
except Exception as e:
    # sys.path.insert(0, os.path.abspath(".."))
    sys.path.insert(0, "/home/runner/work/AIVA_2022_TrafficDensityApp/AIVA_2022_TrafficDensityApp/")


import src.heat_map as heatmap

img = np.ones((5000,5000,3))
imList = np.ones((5000,5000,3,5))
data = np.ones((3,3))


class TestHeatMap(unittest.TestCase):
    def test_init_heatmap(self):
        hp = heatmap.HeatMap()
        self.assertEquals(hp.res, 1) 
    
    def test_local_heat_map(self):
        hp = heatmap.HeatMap()
        a = hp.local_heat_map(img)
        self.assertGreater(a, 1)

    def test_global_heat_map(self):
        hp = heatmap.HeatMap()
        a = hp.global_heat_map(img)
        self.assertGreater(a, 1)

    def test_create_mask(self):
        hp = heatmap.HeatMap()
        a = hp.create_mask(img)
        self.assertGreater(a, 1)

    def test_run(self):
            hp = heatmap.HeatMap()
            a = hp.run(data)
            self.assertEquals(a, 1)


if __name__ == "__main__":
    unittest.main()

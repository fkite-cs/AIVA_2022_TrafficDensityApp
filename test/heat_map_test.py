import unittest
import sys
import numpy as np
from src.cropmanager import CropImg

try:
    sys.path.insert(0, "..")
    from src.yolo_utils import detect_from_folder
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

    def test_global_heat_map(self):
        hp = heatmap.HeatMap()
        img = np.ones((10,10,3), dtype=np.uint8)
        ci = CropImg(0,(0,0),0,0)
        ci.set_vehicles(np.array([[0,0,0,0]]))
        cl = [ci]
        a = hp.global_heat_map(img, cl, ".")
        self.assertEquals(a, 1)



if __name__ == "__main__":
    unittest.main()

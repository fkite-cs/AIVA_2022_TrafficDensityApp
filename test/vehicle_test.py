import unittest
import sys
import os
import cv2
import numpy as np

try:
    sys.path.insert(0, "..")
    from src.yolo_utils import detect_from_folder
except Exception as e:
    # sys.path.insert(0, os.path.abspath(".."))
    sys.path.insert(0, "/home/runner/work/AIVA_2022_TrafficDensityApp/AIVA_2022_TrafficDensityApp/")

import src.vehicle as vehicle

global_hw = (1,1)
y = 2
x = 3
dx = 4
dy = 5

class TestVehicle(unittest.TestCase):

    def test_init_vehicle(self):
        v = vehicle.Vehicle(global_hw, x, y, dx, dy)
        self.assertEquals(v.global_hw, global_hw)
        self.assertEquals(v.y, y)
        self.assertEquals(v.x, x)
        self.assertEquals(v.dy, dy)
        self.assertEquals(v.dx, dx)
    
    def test_get_bbox(self):
        v = vehicle.Vehicle(global_hw, x, y, dx, dy)
        bbox = v.get_bbox()
        self.assertEquals(bbox, [x,y,dx,dy])

    def test_get_global_coordinates(self):
        v = vehicle.Vehicle(global_hw, x, y, dx, dy)
        gc = v.get_global_coordinates()
        _gc = (global_hw[0] + y, global_hw[1] + x)
        self.assertEquals(gc, _gc)

if __name__ == "__main__":
    unittest.main()

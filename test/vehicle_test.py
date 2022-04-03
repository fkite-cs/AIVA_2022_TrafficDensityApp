import unittest
import sys
import os
import cv2
import numpy as np

try:
    sys.path.insert(0, "..")
    from src.yolo_utils import detect2
except Exception as e:
    # sys.path.insert(0, os.path.abspath(".."))
    sys.path.insert(0, "/home/runner/work/AIVA_2022_TrafficDensityApp/AIVA_2022_TrafficDensityApp/")

import src.vehicle as vehicle

class TestVehicle(unittest.TestCase):


    def test_init_vehicle(self):
        v = vehicle.Vehicle((1,1), 2, 2, 3, 3)
        

if __name__ == "__main__":
    unittest.main()

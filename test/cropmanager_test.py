import unittest
import sys
import numpy as np

try:
    sys.path.insert(0, "..")
    from src.yolo_utils import detect2
except Exception as e:
    # sys.path.insert(0, os.path.abspath(".."))
    sys.path.insert(0, "/home/runner/work/AIVA_2022_TrafficDensityApp/AIVA_2022_TrafficDensityApp/")

import src.cropmanager as cropmanager

n=3
hw=(2,2)
dh=2
dw=3
img = np.ones((5000,5000))

class TestCropManager(unittest.TestCase):
    def test_init_cropmanager(self):
        cm = cropmanager.CropManager()
        self.assertEquals(cm.crop_list, [])
        self.assertEquals(cm.crop_size, 250)
    
    def test_create_crops(self):
        cm = cropmanager.CropManager()
        crop_list = cm.create_crops(img)
        self.assertEquals(crop_list, [5,6,7])

    def test_add_vehicles(self):
        cm = cropmanager.CropManager()
        m = cm.add_vehicles(bboxs=np.ones((3,2,4)))
        self.assertGreater(m, 1)


class TestCropImg(unittest.TestCase):
    def test_init_CropImg(self):
        c = cropmanager.CropImg(n, hw, dh, dw)
        self.assertEquals(c.id, n)
        self.assertEquals(c.hw, hw)
        self.assertEquals(c.dh, dh)
        self.assertEquals(c.dw, dw)
        self.assertEquals(c.vehicles_list, [])

    def test_set_vehicles(self):
        c = cropmanager.CropImg(n, hw, dh, dw)
        bboxslen = c.set_vehicles(np.ones((3,4)))
        self.assertGreater(bboxslen, 1)

    def test_get_crop(self):
        c = cropmanager.CropImg(n, hw, dh, dw)
        a = c.get_crop(img)
        self.assertEquals(a.all(), img[
            hw[0]:hw[0] + dh,
            hw[1]:hw[1] + dw].all())

if __name__ == "__main__":
    unittest.main()

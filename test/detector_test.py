import unittest
import sys
import os
import cv2
import numpy as np

try:
    import src.yolo_utils as yu
    from src.yolo_utils import detect2
except Exception as e:
    sys.path.insert(0, os.path.abspath(".."))
    import src.yolo_utils as yu
    

# def detect_car(model, img_path, imgsz, pt, stride, opt):
#     return [1,2,3]

# def load_model(weights):
#     return None, None , None, None, None

class TestVehicleDetector(unittest.TestCase):

    def test_load_img(self):

        opt = {
            "conf_thres": 0.25,
            "iou_thres": 0.45,
            "classes": None,
            "agnostic_nms": False,
            "max_det": 1000
        }

        weights_path = "best.pt"
        img_path = "imgs/austin1_cropped_2.jpg"
        img = cv2.imread(img_path)
        img = img.astype(np.float32)
        model, imgsz, stride, pt, names = yu.load_model(weights=weights_path, device="cpu")
        preds = yu.detect2(model, "imgs/", imgsz, pt, stride, opt, device="cpu")
        # preds = detect_car(model, "imgs/", imgsz, pt, stride, opt)
        self.assertGreater(len(preds), 0)

if __name__ == "__main__":
    unittest.main()
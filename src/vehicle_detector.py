import cv2
from src.cropmanager import CropManager
from src.detector_base import DetectorBase
from src.heat_map import HeatMap
import pdb

class VehicleDetector():

    def __init__(self, model_type, config) -> None:
        self.cm = CropManager() # 
        self.detector = DetectorBase(model_type, config)
        self.hm = HeatMap()


    def run(self, img):
        """
            img: (5000,5000)
        """
        # pdb.set_trace()
        self.img = img
        self.cm.create_crops(img)
        bbox_list = self.detector.forward(self.cm.get_crops(img))
        self.cm.add_vehicles(bbox_list)
        self.ghm = self.hm.run(self.cm)
        
        for i, c in enumerate(self.cm.crop_list):
            _c = c.draw_global_rectangles(img)
            cv2.imwrite(f"examples/crops/{str(i)}.png", _c)
        return self.ghm
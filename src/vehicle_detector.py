from src.cropmanager import CropManager
from src.detector_base import DetectorBase
from src.heat_map import HeatMap


class VehicleDetector():

    def __init__(self, model_type, config) -> None:
        self.cm = CropManager() # 
        self.detector = DetectorBase(model_type, config)
        self.hm = HeatMap()
       
        
    def run(self, img):
        """
            img: (5000,5000)
        """
        self.cm.create_crops(img)
        bbox_list = self.detector.forward(self.cm.crop_list)
        self.cm.add_vehicles(bbox_list)
        self.ghm = self.hm.run(self.cm)
        return self.ghm
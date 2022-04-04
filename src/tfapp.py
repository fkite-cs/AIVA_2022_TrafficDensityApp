import cv2
import yaml

import src.vehicle_detector as vd

class TFApp():

    def __init__(self, vd_config, model_type) -> None:
        self.vd_config = vd_config
        self.model_type = model_type

    def run(self, img, *arg, **args):
        img = cv2.imread(img)
        vd_instance = vd.VehicleDetector(self.model_type, self.vd_config)
        ghm = vd_instance.run(img)
        
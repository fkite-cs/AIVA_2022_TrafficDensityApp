import torch
import numpy as np
import src.yolo_utils as yu

class DetectorBase():

    def __init__(self, model_type, config_model) -> None:
        """
            yolo
            ssd
            ...
        """
        if model_type == "yolo":
            self.model = YOLO(config_model)
        elif model_type == "ssd":
            self.model = SSD(config_model)

    def forward(self, img):
        return self.model.forward(img)


class ModelInterface():

    def __init__(self, config) -> None:
        self.config = config

    def forward(self, *arg, **args):
        raise NotImplementedError

    

class YOLO(ModelInterface):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        self.__load_model()

    def __load_model(self):
        self.model, self.imgsz, self.stride, self.pt, self.names = \
            yu.load_model(self.config.weights_path, self.config.device)

    def forward(self, img):
        """
            img: (n,h,w,3)
        """
        # return np.random.randint(low=0,high=10, size=(10,5,4))
        return yu.detect_multi(
            self.model, img, self.imgsz, self.pt,
            self.stride, self.config.yolo, device=self.config.device
        )


class SSD(ModelInterface):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config

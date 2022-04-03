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

    def bbox_2_vehicle(self, bbox):
        pass
    

class YOLO(ModelInterface):
    
    def __init__(self, config) -> None:
        super().__init__(config)
        self.__load_model()

    def __load_model(self):
        self.model = yu.load_model(self.config.weights_path, self.config.device)

    def forward(self, img):
        return np.random.randint(low=0,high=10, size=(10,5,4))

class SSD(ModelInterface):

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config

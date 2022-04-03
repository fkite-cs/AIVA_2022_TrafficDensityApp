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
        pass



class ModelInterface():

    def __init__(self) -> None:
        pass

    def forward(self):
        pass

    def bbox_2_vehicle(self, bbox):
        pass
    

class YOLO(ModelInterface):
    
    def __init__(self) -> None:
        super().__init__()


class SSD(ModelInterface):

    def __init__(self) -> None:
        super().__init__()


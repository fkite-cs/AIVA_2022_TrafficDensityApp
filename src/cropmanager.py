class CropManager():

    def __init__(self) -> None:
        print("hola")

    def create_crops(self, img):
        """
            crops: (250,250)
            CropImg
        """
        self.crop_list = None

    def add_vehicles(self, bboxs):
        """
            bboxs: (m,n,4)
                m: number of crops
        """
        for i, c in enumerate(self.crop_list):
            c.set_vehicle(bboxs[i])
        

class CropImg():

    def __init__(self, n, crop, hw) -> None:
        self.id = n
        self.crop = crop
        self.hw = hw

    def set_vehicles(self, bboxs):
        """
            bboxs: (n, 4)
        """
        self.vehicles_list = bboxs
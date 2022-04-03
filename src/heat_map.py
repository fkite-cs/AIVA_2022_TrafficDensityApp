class HeatMap():

    def __init__(self) -> None:
        self.res = 1

    def local_heat_map(self, img):
        """
            img: (250,250)
        """
        return len(img)

    def global_heat_map(self, img_list):
        """
            list_img
            return:
                img (5000,5000)
        """
        return len(img_list)

    def create_mask(self, img):
        """
            return:
                mask: (250,250)
        """
        return len(img)

    def run(self, data):
        """
            data: CropManager
        """
        return 1
import cv2
import numpy as np

from src.vehicle import Vehicle


class CropManager():

    def __init__(self, crop_size=250) -> None:
        #TODO añadir crop_size (?)
        self.crop_list = []
        self.crop_size = crop_size

    def create_crops(self, img):
        """
            crops: (250,250)
            CropImg
        """
        #TODO
        imgheight = img.shape[0]
        imgwidth = img.shape[1]
        N = self.crop_size
        croped_list = []
        n = 0
        for y in range(0, imgheight, N):
            for x in range(0, imgwidth, N):
                croped_list.append(CropImg(n, (y, x), N, N))
                n += 1

        self.crop_list = np.array(croped_list)
        return self.crop_list

    def add_vehicles(self, bboxs):
        """
            bboxs: (m,n,4)
                m: number of crops
        """
        for i in range(len(self.crop_list)):
            self.crop_list[i].set_vehicles(bboxs[i])

        return len(bboxs)


    def get_crops(self, img):
        crop_list = []
        for i, ci in enumerate(self.crop_list):
            crop_list.append(ci.get_crop(img))
        return crop_list


class CropImg():

    def __init__(self, n, hw, dh, dw) -> None:
        self.id = n
        self.hw = hw
        self.dh = dh
        self.dw = dw
        self.vehicles_list = []

    def set_vehicles(self, bboxs):
        """
            bboxs: (n, 4)
        """
        for i in range(len(bboxs)):
            bb = bboxs[i]
            self.vehicles_list.append(
                Vehicle(self.hw, *bb)
            )
        return len(bboxs)

    def get_crop(self, img):
        return img[
            self.hw[0]:self.hw[0] + self.dh,
            self.hw[1]:self.hw[1] + self.dw
        ]

    def get_result_crop(self, img):
        c = self.get_crop(img)
        c = self.draw_rectangles(c, self.vehicles_list)
        return c

    def draw_rectangles(self, img, pred):
        img = img.astype(np.uint8)
        for _, v in enumerate(pred):
        # for *xywh, conf, cls in reversed(pred):
            xywh = [v.x, v.y, v.dx, v.dy]
            cv2.rectangle(
                img,
                (int(xywh[0]), int(xywh[1])),
                (int(xywh[0] + xywh[2]), int(xywh[1] + xywh[3])),
                (255,0,0),
                thickness=1,
                lineType=cv2.LINE_AA
            )
        return img
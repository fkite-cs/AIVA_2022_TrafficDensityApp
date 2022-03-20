import cv2
import sys
import torch
import numpy as np
import src.yolo_utils as yu

YOLOV5_PATH = "./yolov5"
sys.path.insert(0,YOLOV5_PATH)

opt = {
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "classes": None,
    "agnostic_nms": False,
    "max_det": 1000
}


if __name__ == "__main__":
    weights_path = "best.pt"
    img_path = "imgs/austin1_cropped_2.jpg"
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    print(img.shape)
    model, imgsz, stride, pt, names = yu.load_model(weights=weights_path)
    # pred = yu.detect2(model, img, imgsz, pt, stride, opt)
    preds = yu.detect2(model, "imgs/", imgsz, pt, stride, opt)
    print(">>>>>>> ", preds[0])
import cv2
import sys
import torch
import numpy as np
import src.yolo_utils as yu
import pdb
import time

YOLOV5_PATH = "./yolov5"
sys.path.insert(0,YOLOV5_PATH)

opt = {
    "conf_thres": 0.25,
    "iou_thres": 0.45,
    "classes": None,
    "agnostic_nms": False,
    "max_det": 1000
}

def draw_rectangles(img, pred):
    img = img.astype(np.uint8)
    for *xywh, in reversed(pred):
    # for *xywh, conf, cls in reversed(pred):
        cv2.rectangle(
            img,
            (int(xywh[0]), int(xywh[1])),
            (int(xywh[0] + xywh[2]), int(xywh[1] + xywh[3])),
            (255,0,0),
            thickness=1,
            lineType=cv2.LINE_AA
        )
    cv2.imshow("1", img), cv2.waitKey(0)


if __name__ == "__main__":
    weights_path = "best.pt"
    img_path = "imgs/austin1_cropped.jpg"
    img = cv2.imread(img_path)
    img = img.astype(np.float32)
    img_path = "imgs/austin1_cropped_2.jpg"
    img = np.array([
        img, cv2.imread(img_path).astype(np.float32),
        cv2.imread("examples/aa.png").astype(np.float32)
    ])
    # img = np.array([img])
    model, imgsz, stride, pt, names = yu.load_model(weights=weights_path)
    pt = True
    t0 = time.time()
    preds1 = yu.detect_multi(model, img, imgsz, pt, stride, opt)
    print("time: ", time.time() - t0)
    # preds2 = yu.detect_from_folder(model, "imgs/", imgsz, pt, stride, opt)
    # pdb.set_trace()
    for i in range(len(preds1)):
        print(img[i].shape)
        print(preds1[i].shape)
        draw_rectangles(img[i], pred=preds1[i].cpu().numpy())
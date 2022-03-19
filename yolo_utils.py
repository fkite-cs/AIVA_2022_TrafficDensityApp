import cv2
import torch
import numpy as np
from numpy import random
import sys
import pdb

YOLOV5_PATH = "./yolov5"
sys.path.insert(0,YOLOV5_PATH)


from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from yolov5.utils.datasets import LoadImages

def pre_process_image(img0, imgsz, stride):
    img = letterbox(img0, imgsz, stride)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img)
    return img, img0

def load_model(weights, device=0, imgsz=416):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    return model, imgsz, stride, pt, names

def detect(model, im, imgsz, pt, stride, opt, device=0):
    im, img0 = pre_process_image(im, imgsz, stride)
    im = torch.from_numpy(im).to(device)
    im /= 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)
    pred = model(im, augment=False, visualize=False) # 1 3 96 416 | 0 0.98411
    pred = non_max_suppression(
        pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"], agnostic=opt["agnostic_nms"]
        , max_det=opt["max_det"]
    )
    print(">>>>>>>> ", pred[0].shape)
    for i, det in enumerate(pred):
        if len (det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
    return pred

def detect2(model, im, imgsz, pt, stride, opt, device=0):
    dataset = LoadImages(im, img_size=imgsz, stride=stride, auto=pt)
    path, im, im0s, vid_cap, s = next(iter(dataset))
    im = im.astype(np.float32)
    im = torch.from_numpy(im).to(device)
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=False, visualize=False) # 1 3 96 416 | 0 0.98411
    pred = non_max_suppression(
        pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"], agnostic=opt["agnostic_nms"]
        , max_det=opt["max_det"]
    )
    print(">>>>>>" ,len(pred))
    for i, det in enumerate(pred):
        if len (det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
    return pred
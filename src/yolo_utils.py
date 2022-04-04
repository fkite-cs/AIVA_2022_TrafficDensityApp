import cv2
import torch
import numpy as np
from numpy import random
import sys
import pdb
import os

YOLOV5_PATH = "./yolov5"
sys.path.insert(0,YOLOV5_PATH)
# sys.path.insert(0, os.path.abspath(".."))

from yolov5.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
from yolov5.utils.datasets import LoadImages

def pre_process_image(img0, imgsz, stride, auto):
    img, ratio, (dw, dh) = letterbox(img0, imgsz, stride, auto=auto, scaleFill=True)
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)
    return img, img0, ratio, dw, dh

def load_model(weights, device=0, imgsz=416):
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)
    return model, imgsz, stride, pt, names

def detect_single(model, im, imgsz, pt, stride, opt, device=0):
    im, img0, _, _, _ = pre_process_image(im, imgsz, stride, auto=pt)
    im = im.astype(np.float32)
    im = torch.from_numpy(im).to(device)
    im /= 255.0
    if im.ndimension() == 3:
        im = im.unsqueeze(0)
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(
        pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"],
        agnostic=opt["agnostic_nms"], max_det=opt["max_det"]
    )
    for i, det in enumerate(pred):
        if len (det):
            if pt:
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            else:
                det[:, :4] = scale_coords_ratio(im.shape[2:], det[:, :4], img0.shape).round()
    return pred

def detect_multi(model, im_list, imgsz, pt, stride, opt, device=0):
    batch = len(im_list)
    preds = []
    for i in range(batch):
        im, img0, _, _, _ = pre_process_image(im_list[i], imgsz, stride, auto=pt)
        im = \
            torch.from_numpy(im.astype(np.float32)).to(device) / 255.
        im = im.unsqueeze(0)
        img0 = torch.from_numpy(img0).to(device)
        pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(
            pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"],
            agnostic=opt["agnostic_nms"], max_det=opt["max_det"]
        )
        for i, det in enumerate(pred):
            if len (det):
                if pt:
                    det[:, :4] = scale_coords(im[i].shape[1:], det[:, :4], img0.shape).round()
                else:
                    det[:, :4] = scale_coords_ratio(im[i].shape[1:], det[:, :4], img0.shape).round()
        pred = pred[0]
        pred = xyxy2xywh(pred[:, :4])
        preds.append(pred)
    return preds

def detect_from_folder(model, im, imgsz, pt, stride, opt, device=0):
    dataset = LoadImages(im, img_size=imgsz, stride=stride, auto=pt)
    path, im, im0s, vid_cap, s = next(iter(dataset))
    im = im.astype(np.float32)
    im = torch.from_numpy(im).to(device)
    im /= 255
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    pred = model(im, augment=False, visualize=False)
    pred = non_max_suppression(
        pred, opt["conf_thres"], opt["iou_thres"], classes=opt["classes"],
        agnostic=opt["agnostic_nms"], max_det=opt["max_det"]
    )
    for i, det in enumerate(pred):
        if len (det):
            if pt:
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0s.shape).round()
            else:
                det[:, :4] = scale_coords_ratio(im.shape[2:], det[:, :4], im0s.shape).round()
    return pred


def scale_coords_ratio(img1_shape, coords, img0_shape): # divide nueva / original
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain_h, gain_w = img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]  # gain  = old / new
    gain = min(gain_h, gain_w)
    pads = (img1_shape[1] - img0_shape[1] * gain_w) / 2, (img1_shape[0] - img0_shape[0] * gain_h) / 2
    coords[:, [0, 2]] -= pads[0]  # x padding
    coords[:, [1, 3]] -= pads[1]  # y padding
    coords[:, [0,2]] /= gain_w
    coords[:, [1,3]] /= gain_h
    clip_coords(coords, img0_shape)
    return coords

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3].clamp_(0, img_shape[0])  # y2

def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 2] = x[:, 2] - x[:, 0]  # width
    y[:, 3] = x[:, 3] - x[:, 1]  # height
    return y
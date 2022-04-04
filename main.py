from src.tfapp import TFApp 
from types import SimpleNamespace

if __name__ == "__main__":
    model_type = "yolo"
    opt = {
        "conf_thres": 0.25,
        "iou_thres": 0.45,
        "classes": None,
        "agnostic_nms": False,
        "max_det": 1000
    }

    config = {
        "weights_path": "best.pt",
        "device": "cpu",
        "yolo": opt
    }

    config = SimpleNamespace(**config)
    app = TFApp(model_type=model_type, vd_config=config)

    img_path = "examples/austin1.tif"

    app.run(img_path)
    print("finish :)")
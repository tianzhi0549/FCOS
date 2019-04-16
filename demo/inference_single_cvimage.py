#!/2019/04:15 18:00
#!/2019/04:16 01:02
#!/2019/04:16 21:02

## A simple class FCOS, mainly to `inference_single_cvimage`.
## 创建一个简单的类 FCOS，用于对单张图像进行推理，减少对 maskrcnn_benchmark 的依赖。

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, curdir + "/..")
sys.path.insert(0, curdir)

import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from torchvision import transforms as T
from maskrcnn_benchmark.structures.image_list import to_image_list

import argparse


def create_colors(len=1):
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len)]
    colors = [(int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255)) for rgba in colors]
    return colors


def parse_args():
    parser = argparse.ArgumentParser(description="FCOS Object Detection Demo on OpenCV Image")
    parser.add_argument(
        "--config-file",
        default="../configs/fcos/fcos_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight-file",
        default="../models/FCOS_R_50_FPN_1x.pth",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Minimum score for the prediction to be shown",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device, default cuda:0",
    )

    parser.add_argument(
        "--min-image-size",
        type=int,
        default=400,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )

    parser.add_argument(
        "--image",
        type=str,
        default="man_dog.jpg",
        help="your test image path",
    )

    args = parser.parse_args()
    return args


## A simple class FCOS, main to `inference_single_cvimage`.
class FCOS():
    coco_names = [
    '__background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
    'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
    'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet',
    'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    ]
    coco_colors = create_colors(len(coco_names))

    def __init__(self, cfg, min_image_size=400):
        self.device = cfg.MODEL.DEVICE
        self.model = None
        self.cfg = cfg
        self.transform = self.build_transform(min_image_size)
        self.build_and_load_model()

    def build_and_load_model(self):
        cfg = self.cfg
        model = build_detection_model(cfg)
        model.to(cfg.MODEL.DEVICE)
        checkpointer = DetectronCheckpointer(cfg, model, save_dir = cfg.OUTPUT_DIR)
        _ = checkpointer.load(cfg.MODEL.WEIGHT )
        model.eval()
        self.model = model

    def build_transform(self, min_image_size=400):
        cfg = self.cfg
        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )
        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(min_image_size),
                T.ToTensor(),
                T.Lambda(lambda x: x * 255),
                normalize_transform,
            ]
        )
        return transform

    def inference_single_cvimage(self, img, verbose=True, score_th=0.2):
        nh, nw = img.shape[:2]
        image = self.transform(img)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)

        ## compute prediction
        with torch.no_grad():
            predictions = self.model(image_list)

        prediction = predictions[0].to("cpu")

        # reshape prediction (a BoxList) into the original image size
        prediction = prediction.resize((nw, nh)).convert("xywh")

        ##  sort by scores, and draw bbox
        scores = prediction.get_field("scores")
        keep = torch.nonzero(scores > score_th).squeeze(1)
        prediction = prediction[keep]
        scores = prediction.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        prediction = prediction[idx]

        ## coco_eval.py/prepare_for_coco_detection
        scores = prediction.get_field("scores").tolist()
        labels = prediction.get_field("labels").tolist()
        bboxes = prediction.bbox.numpy().astype(np.int32).tolist()

        if verbose == True:
            canvas = img.copy()
            for bbox, score, label in zip(bboxes, scores, labels):
                x,y,w,h = bbox
                if label < len(self.coco_names):
                    color = self.coco_colors[label]
                    name = self.coco_names[label]
                    caption = "#{} {} {:.3f}".format(label, name, score)
                else:
                    color = (0, 255, 0)
                    caption = "#{}, s({:.3f})".format(label, score)
                _=cv2.rectangle(canvas, (x,y), (x+w, y+h), color, 2, cv2.LINE_AA)
                _=cv2.putText(canvas, caption, (x+5, y+20), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("FCOS", canvas)
            cv2.waitKey();cv2.destroyAllWindows()

        return (bboxes, scores, labels)


if __name__ == "__main__":
    args = parse_args()
    score_th = args.confidence_threshold
    min_image_size = args.min_image_size

    # update cfg
    cfg.merge_from_file(args.config_file)
    cfg.MODEL.DEVICE = args.device
    cfg.MODEL.WEIGHT = args.weight_file
    cfg.TEST.IMS_PER_BATCH = 1  # only test single image
    cfg.freeze()

    # read the image
    img = cv2.imread(args.image)

    # create FCOS object, and infer on the single image
    fcos = FCOS(cfg, min_image_size)
    res =  fcos.inference_single_cvimage(img, verbose=True, score_th = score_th)
    print(res)

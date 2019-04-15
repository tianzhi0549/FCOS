#!/2019/04:15 18:00
#!/2019/04:16 01:02

## A simple class FCOS, mainly to `inference_single_cvimage`.
## 创建一个简单的类 FCOS，用于对单张图像进行推理，减少对 maskrcnn_benchmark 的依赖。

import os, sys
curdir = os.path.dirname(os.path.abspath(__file__)) 
sys.path.insert(0, curdir + "/..")

import cv2
import numpy as np

import torch
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from torchvision import transforms as T
from maskrcnn_benchmark.structures.image_list import to_image_list

## A simple class FCOS, main to `inference_single_cvimage`.
class FCOS():
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
                _=cv2.rectangle(canvas, (x,y), (x+w, y+h), (0, 255, 0), 2, cv2.LINE_AA)
                text = "cid({}), s({:.3f})".format(label, score)
                _=cv2.putText(canvas, text, (x+5, y+20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow("FCOS", canvas)
            cv2.waitKey();cv2.destroyAllWindows()

        return (bboxes, scores, labels)


if __name__ == "__main__":
    fpath = "man_dog.jpg"
    device = "cuda:0"
    cpu_device = torch.device("cpu")
    config_file = r"D:\Projects\Python\201904\FCOS\FCOS-master\configs\fcos\fcos_R_50_FPN_1x.yaml"
    weight_file  = r"D:\Projects\Python\201904\FCOS\FCOS-master\models\FCOS_R_50_FPN_1x.pth"

    img = cv2.imread(fpath)
    
    cfg.merge_from_file(config_file)
    cfg.MODEL.DEVICE = device
    cfg.MODEL.WEIGHT = weight_file
    cfg.TEST.IMS_PER_BATCH = 1 # only test single image
    cfg.freeze()
    fcos = FCOS(cfg)

    for i in range(4):
        score_th = i*0.1 + 0.2
        res =  fcos.inference_single_cvimage(img, verbose=True, score_th = score_th)
        print(res)


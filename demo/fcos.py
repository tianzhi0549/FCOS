import cv2, os
import torch
from maskrcnn_benchmark.config import cfg as base_cfg
from torchvision import transforms as T
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.bounding_box import BoxList


_MODEL_NAMES_TO_URLS = {
    "fcos_R_50_FPN_1x": "https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download#fcos_R_50_FPN_1x.pth"
}


class FCOS(object):
    CATEGORIES = [
        "__background",
        "person",
        "bicycle",
        "car",
        "motorcycle",
        "airplane",
        "bus",
        "train",
        "truck",
        "boat",
        "traffic light",
        "fire hydrant",
        "stop sign",
        "parking meter",
        "bench",
        "bird",
        "cat",
        "dog",
        "horse",
        "sheep",
        "cow",
        "elephant",
        "bear",
        "zebra",
        "giraffe",
        "backpack",
        "umbrella",
        "handbag",
        "tie",
        "suitcase",
        "frisbee",
        "skis",
        "snowboard",
        "sports ball",
        "kite",
        "baseball bat",
        "baseball glove",
        "skateboard",
        "surfboard",
        "tennis racket",
        "bottle",
        "wine glass",
        "cup",
        "fork",
        "knife",
        "spoon",
        "bowl",
        "banana",
        "apple",
        "sandwich",
        "orange",
        "broccoli",
        "carrot",
        "hot dog",
        "pizza",
        "donut",
        "cake",
        "chair",
        "couch",
        "potted plant",
        "bed",
        "dining table",
        "toilet",
        "tv",
        "laptop",
        "mouse",
        "remote",
        "keyboard",
        "cell phone",
        "microwave",
        "oven",
        "toaster",
        "sink",
        "refrigerator",
        "book",
        "clock",
        "vase",
        "scissors",
        "teddy bear",
        "hair drier",
        "toothbrush",
    ]

    def __init__(
            self,
            model_name="fcos_R_50_FPN_1x",
            nms_thresh=0.6,
            cpu_only=False
    ):
        root_dir = os.path.dirname(os.path.dirname(
            os.path.abspath(__file__)
        ))
        self.config_files_dir = os.path.join(root_dir, "configs", "fcos")
        self.cfg_name = model_name + ".yaml"

        cfg = base_cfg.clone()
        cfg.merge_from_file(os.path.join(self.config_files_dir, self.cfg_name))
        cfg.MODEL.WEIGHT = _MODEL_NAMES_TO_URLS[model_name]
        cfg.MODEL.FCOS.NMS_TH = nms_thresh
        if cpu_only:
            cfg.MODEL.DEVICE = "cpu"
        else:
            cfg.MODEL.DEVICE = "cuda"

        cfg.freeze()
        self.cfg = cfg

        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)

        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.transforms = self.build_transform()
        self.cpu_device = torch.device("cpu")

    def detect(self, im, min_confidence=0.4):
        '''
        :param im (np.ndarray): an image as returned by OpenCV
        :return:
        '''
        image = self.transforms(im)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        predictions = [o.to(self.cpu_device) for o in predictions]

        assert len(predictions) == 1
        predictions = predictions[0]

        predictions = self.select_top_predictions(predictions, min_confidence)
        return self._bbox_list_to_py_bbox_list(predictions)

    def _bbox_list_to_py_bbox_list(self, predictions):
        bboxes = predictions.bbox

        results = []
        for i in range(len(predictions)):
            bbox = [float(_) for _ in bboxes[i]]
            label_id = int(predictions.get_field("labels")[i])
            label_name = self.CATEGORIES[label_id]
            score = float(predictions.get_field("scores")[i])

            results.append({
                "box": bbox,
                "score": score,
                "label_name": label_name,
                "label_id": label_id
            })

        return results

    def _py_bbox_list_to_bbox_list(self, py_bbox_list, im_size):
        '''
        :param py_bbox_list:
        :param im_size: (w, h)
        :return:
        '''
        bboxes = []
        labels = []
        scores = []
        for item in py_bbox_list:
            bboxes.append(item["box"])
            labels.append(item["label_id"])
            scores.append(item["score"])

        box_list = BoxList(torch.tensor(bboxes, dtype=torch.float32), im_size)
        box_list.add_field("labels", torch.tensor(labels, dtype=torch.long))
        box_list.add_field("scores", torch.tensor(scores, dtype=torch.float32))

        return box_list

    def select_top_predictions(self, predictions, confidence):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        if isinstance(confidence, float):
            thresholds = confidence
        else:
            thresholds = confidence[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def show_bboxes(self, im, bbox_results):
        bbox_list = self._py_bbox_list_to_bbox_list(
            bbox_results,
            (im.shape[1], im.shape[0])
        )

        im_with_bboxes = self.overlay_boxes(im, bbox_list)
        im_with_bboxes = self.overlay_class_names(im_with_bboxes, bbox_list)
        cv2.imshow("Detections", im_with_bboxes)
        cv2.waitKey()

        return im_with_bboxes

    def overlay_boxes(self, image, predictions):
        """
        Adds the predicted boxes on top of the image

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `labels`.
        """
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = self.compute_colors_for_labels(labels).tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(
                image, tuple(top_left), tuple(bottom_right), tuple(color), 2
            )

        return image

    def overlay_class_names(self, image, predictions):
        """
        Adds detected class names and scores in the positions defined by the
        top-left corner of the predicted bounding box

        Arguments:
            image (np.ndarray): an image as returned by OpenCV
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores` and `labels`.
        """
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.CATEGORIES[i] for i in labels]
        boxes = predictions.bbox

        template = "{}: {:.2f}"
        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            s = template.format(label, score)
            cv2.putText(
                image, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1
            )

        return image

    def compute_colors_for_labels(self, labels):
        """
        Simple function that adds fixed colors depending on the class
        """
        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8")
        return colors

    def list_available_models(self):
        for model_name, _ in _MODEL_NAMES_TO_URLS.items():
            print(model_name)


if __name__ == "__main__":
    root_dir = os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))
    fcos = FCOS(
        model_name="fcos_R_50_FPN_1x",
        nms_thresh=0.6,
        cpu_only=False  # if you do not have GPUs, please set cpu_only as False
    )
    im = cv2.imread(root_dir + "/demo/images/COCO_val2014_000000000885.jpg")
    bbox_results = fcos.detect(im)
    print(bbox_results)

    fcos.show_bboxes(im, bbox_results)

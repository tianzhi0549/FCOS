import cv2, os
import torch
from fcos_core.config import cfg as base_cfg
from torchvision import transforms as T
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.structures.image_list import to_image_list
from fcos_core.structures.bounding_box import BoxList


_MODEL_NAMES_TO_INFO_ = {
    "fcos_R_50_FPN_1x": {
        "url": "https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download#fcos_R_50_FPN_1x.pth",
        "best_min_confidence": [
            0.49211737513542175, 0.49340692162513733, 0.510103702545166,
            0.4707475006580353, 0.5197340250015259, 0.5007652044296265,
            0.5611110329627991, 0.4639902412891388, 0.4778415560722351,
            0.43332818150520325, 0.6180170178413391, 0.5248752236366272,
            0.5437473654747009, 0.5153843760490417, 0.4194680452346802,
            0.5640717148780823, 0.5087228417396545, 0.5021755695343018,
            0.5307778716087341, 0.4920770823955536, 0.5202335119247437,
            0.5715234279632568, 0.5089765191078186, 0.5422378778457642,
            0.45138806104660034, 0.49631351232528687, 0.4388565421104431,
            0.47193753719329834, 0.47037890553474426, 0.4791252017021179,
            0.45699411630630493, 0.48658522963523865, 0.4580649137496948,
            0.4603237509727478, 0.5243804454803467, 0.5235602855682373,
            0.48501554131507874, 0.5173789858818054, 0.4978085160255432,
            0.4626562297344208, 0.48144686222076416, 0.4889853894710541,
            0.4749937951564789, 0.42273756861686707, 0.47836390137672424,
            0.48752328753471375, 0.44069987535476685, 0.4241463541984558,
            0.5228247046470642, 0.4834112524986267, 0.4538525640964508,
            0.4730372428894043, 0.471712201833725, 0.5180512070655823,
            0.4671719968318939, 0.46602892875671387, 0.47536996006965637,
            0.487352192401886, 0.4771934747695923, 0.45533207058906555,
            0.43941256403923035, 0.5910647511482239, 0.554875910282135,
            0.49752360582351685, 0.6263655424118042, 0.4964958727359772,
            0.5542593002319336, 0.5049241185188293, 0.5306999087333679,
            0.5279538035392761, 0.5708096623420715, 0.524990975856781,
            0.5187852382659912, 0.41242220997810364, 0.5409807562828064,
            0.48504579067230225, 0.47305455803871155, 0.4814004898071289,
            0.42680642008781433, 0.4143834114074707
        ]
    }
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
        root_dir = os.path.dirname(os.path.abspath(__file__))
        self.config_files_dir = os.path.join(root_dir, "configs")
        self.cfg_name = model_name + ".yaml"

        cfg = base_cfg.clone()
        cfg.merge_from_file(os.path.join(self.config_files_dir, self.cfg_name))
        cfg.MODEL.WEIGHT = _MODEL_NAMES_TO_INFO_[model_name]["url"]
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
        self.model_name = model_name

    def detect(self, im, min_confidence=None):
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

        if min_confidence is None:
            min_confidence = _MODEL_NAMES_TO_INFO_[self.model_name]["best_min_confidence"]

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
            confidence = scores.new_tensor(confidence)
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
        im = im.copy()
        bbox_list = self._py_bbox_list_to_bbox_list(
            bbox_results,
            (im.shape[1], im.shape[0])
        )

        self.overlay_boxes(im, bbox_list)
        self.overlay_class_names(im, bbox_list)
        cv2.imshow("Detections", im)
        cv2.waitKey()

        return im

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
            cv2.rectangle(
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
        for model_name, _ in _MODEL_NAMES_TO_INFO_.items():
            print(model_name)


if __name__ == "__main__":
    import skimage.io as io
    import argparse

    parser = argparse.ArgumentParser(description="FCOS Object Detector")
    parser.add_argument(
        "input_image",
        help="path or url to an input image",
    )

    args = parser.parse_args()

    fcos = FCOS(
        model_name="fcos_R_50_FPN_1x",
        nms_thresh=0.6,
        cpu_only=False  # if you do not have GPUs, please set cpu_only as False
    )

    im = io.imread(args.input_image)
    assert im.shape[-1] == 3, "only 3-channel images are supported"
    # convert from RGB to BGR because fcos assumes the BRG input image
    im = im[..., ::-1].copy()

    # resize image to have its shorter size == 800
    f = 800.0 / float(min(im.shape[:2]))
    im = cv2.resize(im, (0, 0), fx=f, fy=f)

    bbox_results = fcos.detect(im)

    fcos.show_bboxes(im, bbox_results)

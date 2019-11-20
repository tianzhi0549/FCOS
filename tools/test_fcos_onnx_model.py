"""
An example:
python tools/test_fcos_onnx_model.py --onnx-model fcos_imprv_R_50_FPN_1x.onnx --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml TEST.IMS_PER_BATCH 1 INPUT.MIN_SIZE_TEST 384 DATALOADER.NUM_WORKERS 0

"""
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from torch import nn
import onnx
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.engine.inference import inference
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
from fcos_core.modeling.rpn.fcos.inference import make_fcos_postprocessor
import caffe2.python.onnx.backend as backend
import numpy as np
from fcos_core.structures.image_list import to_image_list


class ONNX_FCOS(nn.Module):
    def __init__(self, onnx_model_path, cfg):
        super(ONNX_FCOS, self).__init__()
        self.onnx_model = backend.prepare(
            onnx.load(onnx_model_path),
            device=cfg.MODEL.DEVICE.upper()
        )
        self.postprocessing = make_fcos_postprocessor(cfg)
        self.cfg = cfg
        self.fpn_strides = cfg.MODEL.FCOS.FPN_STRIDES

    def forward(self, images):
        outputs = self.onnx_model.run(images.tensors.cpu().numpy())
        outputs = [torch.from_numpy(o).to(self.cfg.MODEL.DEVICE) for o in outputs]
        num_outputs = len(outputs) // 3
        logits = outputs[:num_outputs]
        bbox_reg = outputs[num_outputs:2 * num_outputs]
        centerness = outputs[2 * num_outputs:]

        locations = self.compute_locations(logits)
        boxes = self.postprocessing(locations, logits, bbox_reg, centerness, images.image_sizes)
        return boxes

    def compute_locations(self, features):
        locations = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            locations_per_level = self.compute_locations_per_level(
                h, w, self.fpn_strides[level],
                feature.device
            )
            locations.append(locations_per_level)
        return locations

    def compute_locations_per_level(self, h, w, stride, device):
        shifts_x = torch.arange(
            0, w * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shifts_y = torch.arange(
            0, h * stride, step=stride,
            dtype=torch.float32, device=device
        )
        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        locations = torch.stack((shift_x, shift_y), dim=1) + stride // 2
        return locations


def main():
    parser = argparse.ArgumentParser(description="Test onnx models of FCOS")
    parser.add_argument(
        "--config-file",
        default="/private/home/fmassa/github/detectron.pytorch_v2/configs/e2e_faster_rcnn_R_50_C4_1x_caffe2.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--onnx-model",
        default="fcos_imprv_R_50_FPN_1x.onnx",
        metavar="FILE",
        help="path to the onnx model",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    save_dir = ""
    logger = setup_logger("fcos_core", save_dir, get_rank())
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = ONNX_FCOS(args.onnx_model, cfg)
    model.to(cfg.MODEL.DEVICE)

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(cfg, is_train=False, is_distributed=False)
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        inference(
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.FCOS_ON or cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()


if __name__ == "__main__":
    main()

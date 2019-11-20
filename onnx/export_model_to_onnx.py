"""
Please make sure you are using pytorch >= 1.4.0.
A working example to export the R-50 based FCOS model:
python onnx/export_model_to_onnx.py \
    --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
    MODEL.WEIGHT FCOS_imprv_R_50_FPN_1x.pth

"""
from fcos_core.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os

import torch
from fcos_core.config import cfg
from fcos_core.data import make_data_loader
from fcos_core.engine.inference import inference
from fcos_core.modeling.detector import build_detection_model
from fcos_core.utils.checkpoint import DetectronCheckpointer
from fcos_core.utils.collect_env import collect_env_info
from fcos_core.utils.comm import synchronize, get_rank
from fcos_core.utils.logger import setup_logger
from fcos_core.utils.miscellaneous import mkdir
from collections import OrderedDict


def main():
    parser = argparse.ArgumentParser(description="Export model to the onnx format")
    parser.add_argument(
        "--config-file",
        default="configs/fcos/fcos_imprv_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--output",
        default="fcos.onnx",
        metavar="FILE",
        help="path to the output onnx file",
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

    assert cfg.MODEL.FCOS_ON, "This script is only tested for the detector FCOS."

    save_dir = ""
    logger = setup_logger("fcos_core", save_dir, get_rank())
    logger.info(cfg)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    model = build_detection_model(cfg)
    model.to(cfg.MODEL.DEVICE)

    output_dir = cfg.OUTPUT_DIR
    checkpointer = DetectronCheckpointer(cfg, model, save_dir=output_dir)
    _ = checkpointer.load(cfg.MODEL.WEIGHT)

    onnx_model = torch.nn.Sequential(OrderedDict([
        ('backbone', model.backbone),
        ('heads', model.rpn.head),
    ]))

    input_names = ["input_image"]
    dummy_input = torch.zeros((1, 3, 800, 1216)).to(cfg.MODEL.DEVICE)
    output_names = []
    for l in range(len(cfg.MODEL.FCOS.FPN_STRIDES)):
        fpn_name = "P{}/".format(3 + l)
        output_names.extend([
            fpn_name + "logits",
            fpn_name + "bbox_reg",
            fpn_name + "centerness"
        ])

    torch.onnx.export(
        onnx_model, dummy_input,
        args.output, verbose=True,
        input_names=input_names,
        output_names=output_names,
        keep_initializers_as_inputs=True
    )

    logger.info("Done. The onnx model is saved into {}.".format(args.output))


if __name__ == "__main__":
    main()

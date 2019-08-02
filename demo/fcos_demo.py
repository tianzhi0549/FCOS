# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import argparse
import cv2, os

from maskrcnn_benchmark.config import cfg
from predictor import COCODemo

import time


def main():
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
    parser.add_argument(
        "--config-file",
        default="configs/fcos/fcos_R_50_FPN_1x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weights",
        default="FCOS_R_50_FPN_1x.pth",
        metavar="FILE",
        help="path to the trained model",
    )
    parser.add_argument(
        "--images-dir",
        default="demo/images",
        metavar="DIR",
        help="path to demo images directory",
    )
    parser.add_argument(
        "--min-image-size",
        type=int,
        default=800,
        help="Smallest size of the image to feed to the model. "
            "Model was trained with 800, which gives best results",
    )
    parser.add_argument(
        "opts",
        help="Modify model config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    # load config from file and command-line arguments
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.MODEL.WEIGHT = args.weights

    cfg.freeze()

    # The following per-class thresholds are computed by maximizing
    # per-class f-measure in their precision-recall curve.
    # Please see compute_thresholds_for_classes() in coco_eval.py for details.
    thresholds_for_classes = [
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

    demo_im_names = os.listdir(args.images_dir)

    # prepare object that handles inference plus adds predictions on top of image
    coco_demo = COCODemo(
        cfg,
        confidence_thresholds_for_classes=thresholds_for_classes,
        min_image_size=args.min_image_size
    )

    for im_name in demo_im_names:
        img = cv2.imread(os.path.join(args.images_dir, im_name))
        if img is None:
            continue
        start_time = time.time()
        composite = coco_demo.run_on_opencv_image(img)
        print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
        cv2.imshow(im_name, composite)
    print("Press any keys to exit ...")
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


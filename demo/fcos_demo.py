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
        0.23860901594161987, 0.24108672142028809, 0.2470853328704834,
        0.2316885143518448, 0.2708061933517456, 0.23173952102661133,
        0.31990334391593933, 0.21302376687526703, 0.20151866972446442,
        0.20928964018821716, 0.3793887197971344, 0.2715213894844055,
        0.2836397588253021, 0.26449233293533325, 0.1728038638830185,
        0.314998596906662, 0.28575003147125244, 0.28987520933151245,
        0.2727000117301941, 0.23306897282600403, 0.265937477350235,
        0.32663893699645996, 0.27102580666542053, 0.29177549481391907,
        0.2043062448501587, 0.24331751465797424, 0.20752687752246857,
        0.22951272130012512, 0.22753854095935822, 0.2159966081380844,
        0.1993938684463501, 0.23676514625549316, 0.20982342958450317,
        0.18315598368644714, 0.2489681988954544, 0.24793922901153564,
        0.287187397480011, 0.23045086860656738, 0.2462811917066574,
        0.21191294491291046, 0.22845126688480377, 0.24365000426769257,
        0.22687821090221405, 0.18365581333637238, 0.2035856395959854,
        0.23478077352046967, 0.18431290984153748, 0.18184082210063934,
        0.2708037495613098, 0.2268175482749939, 0.19970566034317017,
        0.21832780539989471, 0.21120598912239075, 0.270445853471756,
        0.189377561211586, 0.2101106345653534, 0.2112293541431427,
        0.23484709858894348, 0.22701986134052277, 0.20732736587524414,
        0.1953316181898117, 0.3237660229206085, 0.3078872859477997,
        0.2881140112876892, 0.38746657967567444, 0.20038367807865143,
        0.28123822808265686, 0.2588447630405426, 0.2796839773654938,
        0.266757994890213, 0.3266656696796417, 0.25759157538414,
        0.2578003704547882, 0.17009201645851135, 0.29051828384399414,
        0.24002137780189514, 0.22378061711788177, 0.26134759187698364,
        0.1730124056339264, 0.1857597529888153
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


# FCOS: Fully Convolutional One-Stage Object Detection

This project hosts the code for implementing the FCOS algorithm for object detection, as presented in our paper:

    FCOS: Fully Convolutional One-Stage Object Detection,
    Tian, Zhi, Chunhua Shen, Hao Chen, and Tong He,
    arXiv preprint arXiv:1904.01355 (2019).

The full paper is available at: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355). 

## Highlights
- **Totally anchor-free:**  FCOS completely avoids the complicated computation related to anchor boxes and all hyper-parameters of anchor boxes.   
- **Memory-efficient:** FCOS uses 2x less training memory footprint than its anchor-based counterpart RetinaNet.
- **Better performance:** Compared to RetinaNet, FCOS achieves better performance under exactly the same training and testing settings.
- **State-of-the-art performance:** Without bells and whistles, FCOS achieves state-of-the-art performances.
It achieves **41.0%** (ResNet-101-FPN) and **42.1%** (ResNeXt-32x8d-101) in AP on coco test-dev.
- **Faster:** FCOS enjoys faster training and inference speed than RetinaNet. For example, with 8 Nvidia V100 GPUs, the training of ResNet-50-FPN based FCOS can be done within 7 hours.

## Required hardware
We use 8 Nvidia V100 GPUs. \
But 4 1080Ti GPUs can also train a fully-fledged ResNet-50-FPN based FCOS since FCOS is memory-efficient.  

## Installation

This FCOS implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as original maskrcnn-benchmark.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.

## Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
        MODEL.WEIGHT models/FCOS_R_50_FPN_1x.pth \
        TEST.IMS_PER_BATCH 4    

Please note that:
1) If your model's name is different, please replace `models/FCOS_R_50_FPN_1x.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/fcos](configs/fcos)) and `MODEL.WEIGHT` to its weights file.

For your convenience, we provide the following trained models (more models are coming soon).

Model | Total training mem (GB) | Multi-scale training | Testing time / im | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:|:--:|:---:
FCOS_R_50_FPN_1x | 29.3 | No | 71ms | 36.6 | 37.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download)
FCOS_R_101_FPN_2x | 44.1 | Yes | 74ms | 40.9 | 41.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/vjL3L0AW7vnhRTo/download)
FCOS_X_101_32x8d_FPN_2x | 72.9 | Yes | 122ms | 42.0 | 42.1 | [download](https://cloudstor.aarnet.edu.au/plus/s/U5myBfGF7MviZ97/download)

[1] *1x means the model is trained for 90K iterations.* \
[2] *2x means the model is trained for 180K iterations.* \
[3] *We report total training memory footprint on all GPUs instead of the memory footprint per GPU as in maskrcnn-benchmark*. \
[4] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..*

## Training

The following command line will train FCOS_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --skip-test \
        --config-file configs/fcos/fcos_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_R_50_FPN_1x
        
Note that:
 
1) If you want to use fewer GPUs, please reduce `--nproc_per_node`. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x.yaml](configs/fcos/fcos_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.
4) Sometimes you may encounter a deadlock with 100% GPUs' usage, which might be a problem of NCCL. Please try `export NCCL_P2P_DISABLE=1` before running the training command line.

## Contributing to the project

Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@article{tian2019fcos,
  title={FCOS: Fully Convolutional One-Stage Object Detection},
  author={Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal={arXiv preprint arXiv:1904.01355},
  year={2019}
}
```


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 

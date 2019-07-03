# FCOS: Fully Convolutional One-Stage Object Detection

This project hosts the code for implementing the FCOS algorithm for object detection, as presented in our paper:

    FCOS: Fully Convolutional One-Stage Object Detection;
    Tian Zhi, Chunhua Shen, Hao Chen, and Tong He;
    arXiv preprint arXiv:1904.01355 (2019).

The full paper is available at: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355). 

## Highlights
- **Totally anchor-free:**  FCOS completely avoids the complicated computation related to anchor boxes and all hyper-parameters of anchor boxes.   
- **Memory-efficient:** FCOS uses 2x less training memory footprint than its anchor-based counterpart RetinaNet.
- **Better performance:** The very simple detector achieves better performance (37.1 vs. 36.8) than Faster R-CNN.
- **Faster training and inference:** With the same hardwares, FCOS also requires less training hours (6.5h vs. 8.8h) and faster inference speed (71ms vs. 126 ms per im) than Faster R-CNN.
- **State-of-the-art performance:** Without bells and whistles, FCOS achieves state-of-the-art performances.
It achieves **41.5%** (ResNet-101-FPN) and **43.2%** (ResNeXt-64x4d-101) in AP on coco test-dev.

## Updates
### 17 May 2019
   - FCOS has been implemented in [mmdetection](https://github.com/open-mmlab/mmdetection). Many thanks to [@yhcao6](https://github.com/yhcao6) and [@hellock](https://github.com/hellock).

## Required hardware
We use 8 Nvidia V100 GPUs. \
But 4 1080Ti GPUs can also train a fully-fledged ResNet-50-FPN based FCOS since FCOS is memory-efficient.  

## Installation

This FCOS implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as original maskrcnn-benchmark.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.

## A quick demo
Once the installation is done, you can follow the below steps to run a quick demo.
    
    # assume that you are under the root directory of this project,
    # and you have activated your virtual environment if needed.
    wget https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download -O FCOS_R_50_FPN_1x.pth
    python demo/fcos_demo.py


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

**ResNe(x)ts:**

*All ResNe(x)t based models are trained with 16 images in a mini-batch and frozen batch normalization (i.e., consistent with models in [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)).*

Model | Total training mem (GB) | Multi-scale training | Testing time / im | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:|:--:|:---:
FCOS_R_50_FPN_1x | 29.3 | No | 71ms | 37.1 | 37.4 | [download](https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download)
FCOS_R_101_FPN_2x | 44.1 | Yes | 74ms | 41.4 | 41.5 | [download](https://cloudstor.aarnet.edu.au/plus/s/vjL3L0AW7vnhRTo/download)
FCOS_X_101_32x8d_FPN_2x | 72.9 | Yes | 122ms | 42.5 | 42.7 | [download](https://cloudstor.aarnet.edu.au/plus/s/U5myBfGF7MviZ97/download)
FCOS_X_101_64x4d_FPN_2x | 77.7 | Yes | 140ms | 43.0 | 43.2 | [download](https://cloudstor.aarnet.edu.au/plus/s/wpwoCi4S8iajFi9/download)

**MobileNets:**

*We update batch normalization for MobileNet based models. If you want to use SyncBN, please install pytorch-nightly.*

Model Training batch size | Multi-scale training | Testing time / im | AP (minival) | Link
--- |:---:|:---:|:---:|:---:
FCOS_syncbn_bs32_c128_MNV2_FPN_1x | 32 | No | 19ms | 30.9 | [download](https://cloudstor.aarnet.edu.au/plus/s/3GKwaxZhDSOlCZ0/download)
FCOS_syncbn_bs32_MNV2_FPN_1x | 32 | No | 59ms | 33.1 | [download](https://cloudstor.aarnet.edu.au/plus/s/OpJtCJLj104i2Yc/download)
FCOS_bn_bs16_MNV2_FPN_1x | 16 | No | 59ms | 31.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/B6BrLAiAEAYQkcy/download)

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *We report total training memory footprint on all GPUs instead of the memory footprint per GPU as in maskrcnn-benchmark*. \
[3] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[4] *Our results have been improved since our initial release. If you want to check out our original results, please checkout commit [f4fd589](https://github.com/tianzhi0549/FCOS/tree/f4fd58966f45e64608c00b072c801de7f86b4f3a)*. \
[5] *`c128` denotes the model has 128 (instead of 256) channels in towers (i.e., `MODEL.RESNETS.BACKBONE_OUT_CHANNELS` in [config](https://github.com/tianzhi0549/FCOS/blob/master/configs/fcos/fcos_syncbn_bs32_c128_MNV2_FPN_1x.yaml#L10)).*
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
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x.yaml](configs/fcos/fcos_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.
4) The link of ImageNet pre-training X-101-64x4d in the code is invalid. Please download the model [here](https://cloudstor.aarnet.edu.au/plus/s/k3ys35075jmU1RP/download).
5) If you want to train FCOS on your own dataset, please follow this instruction [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
## Contributing to the project

Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@article{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  journal =  {arXiv preprint arXiv:1904.01355},
  year    =  {2019}
}
```


## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 

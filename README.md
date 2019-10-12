# FCOS: Fully Convolutional One-Stage Object Detection

This project hosts the code for implementing the FCOS algorithm for object detection, as presented in our paper:

    FCOS: Fully Convolutional One-Stage Object Detection;
    Zhi Tian, Chunhua Shen, Hao Chen, and Tong He;
    In: Proc. Int. Conf. Computer Vision (ICCV), 2019.
    arXiv preprint arXiv:1904.01355 

The full paper is available at: [https://arxiv.org/abs/1904.01355](https://arxiv.org/abs/1904.01355). 

## Highlights
- **Totally anchor-free:**  FCOS completely avoids the complicated computation related to anchor boxes and all hyper-parameters of anchor boxes.   
- **Better performance:** The very simple one-stage detector achieves much better performance (38.7 vs. 36.8 in AP with ResNet-50) than Faster R-CNN. Check out more models and experimental results [here](#models).
- **Faster training and testing:** With the same hardwares and backbone ResNet-50-FPN, FCOS also requires less training hours (6.5h vs. 8.8h) than Faster R-CNN. FCOS also takes 12ms less inference time per image than Faster R-CNN (44ms vs. 56ms).
- **State-of-the-art performance:** Our best model based on ResNeXt-64x4d-101 and deformable convolutions achieves **49.0%** in AP on COCO test-dev (with multi-scale testing).

## Updates
   - New NMS (see [#165](https://github.com/tianzhi0549/FCOS/pull/165)) speeds up ResNe(x)t based models by up to 30% and MobileNet based models by 40%, with exactly the same performance. Check out [here](#models). (12/10/2019)
   - New models with much improved performance are released. The best model achieves **49%** in AP on COCO test-dev with multi-scale testing. (11/09/2019)
   - FCOS with VoVNet backbones is available at [VoVNet-FCOS](https://github.com/vov-net/VoVNet-FCOS). (08/08/2019)
   - A trick of using a small central region of the BBox for training improves AP by nearly 1 point [as shown here](https://github.com/yqyao/FCOS_PLUS). (23/07/2019)
   - FCOS with HRNet backbones is available at [HRNet-FCOS](https://github.com/HRNet/HRNet-FCOS). (03/07/2019)
   - FCOS with AutoML searched FPN (R50, R101, ResNeXt101 and MobileNetV2 backbones) is available at [NAS-FCOS](https://github.com/Lausannen/NAS-FCOS). (30/06/2019)
   - FCOS has been implemented in [mmdetection](https://github.com/open-mmlab/mmdetection). Many thanks to [@yhcao6](https://github.com/yhcao6) and [@hellock](https://github.com/hellock). (17/05/2019)

## Required hardware
We use 8 Nvidia V100 GPUs. \
But 4 1080Ti GPUs can also train a fully-fledged ResNet-50-FPN based FCOS since FCOS is memory-efficient.  

## Installation
#### Testing-only installation 
For users who only want to use FCOS as an object detector in their projects, they can install it by pip. To do so, run:
```
pip install torch  # install pytorch if you do not have it
pip install fcos
# run this command line for a demo 
fcos https://github.com/tianzhi0549/FCOS/raw/master/demo/images/COCO_val2014_000000000885.jpg
```
Please check out [here](fcos/bin/fcos) for the interface usage.

#### For a complete installation 
This FCOS implementation is based on [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark). Therefore the installation is the same as original maskrcnn-benchmark.

Please check [INSTALL.md](INSTALL.md) for installation instructions.
You may also want to see the original [README.md](MASKRCNN_README.md) of maskrcnn-benchmark.

## A quick demo
Once the installation is done, you can follow the below steps to run a quick demo.
    
    # assume that you are under the root directory of this project,
    # and you have activated your virtual environment if needed.
    wget https://cloudstor.aarnet.edu.au/plus/s/ZSAqNJB96hA71Yf/download -O FCOS_imprv_R_50_FPN_1x.pth
    python demo/fcos_demo.py


## Inference
The inference command line on coco minival split:

    python tools/test_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        MODEL.WEIGHT FCOS_imprv_R_50_FPN_1x.pth \
        TEST.IMS_PER_BATCH 4    

Please note that:
1) If your model's name is different, please replace `FCOS_imprv_R_50_FPN_1x.pth` with your own.
2) If you enounter out-of-memory error, please try to reduce `TEST.IMS_PER_BATCH` to 1.
3) If you want to evaluate a different model, please change `--config-file` to its config file (in [configs/fcos](configs/fcos)) and `MODEL.WEIGHT` to its weights file.
4) Multi-GPU inference is available, please refer to [#78](https://github.com/tianzhi0549/FCOS/issues/78#issuecomment-526990989).
5) We improved the postprocess efficiency by using multi-label nms (see [#165](https://github.com/tianzhi0549/FCOS/pull/165)), which saves 18ms on average. The inference metric in the following tables has been updated accordingly.

## Models
For your convenience, we provide the following trained models (more models are coming soon).

**ResNe(x)ts:**

*All ResNe(x)t based models are trained with 16 images in a mini-batch and frozen batch normalization (i.e., consistent with models in [maskrcnn_benchmark](https://github.com/facebookresearch/maskrcnn-benchmark)).*

Model | Multi-scale training | Testing time / im | AP (minival) | Link
--- |:---:|:---:|:---:|:---:
FCOS_imprv_R_50_FPN_1x | No | 44ms | 38.7 | [download](https://cloudstor.aarnet.edu.au/plus/s/ZSAqNJB96hA71Yf/download)
FCOS_imprv_dcnv2_R_50_FPN_1x | No | 54ms | 42.3 | [download](https://cloudstor.aarnet.edu.au/plus/s/plKgHuykjiilzWr/download)
FCOS_imprv_R_101_FPN_2x | Yes | 57ms | 43.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/hTeMuRa4pwtCemq/download)
FCOS_imprv_dcnv2_R_101_FPN_2x | Yes | 73ms | 45.6 | [download](https://cloudstor.aarnet.edu.au/plus/s/xq2Ll7s0hpaQycO/download)
FCOS_imprv_X_101_32x8d_FPN_2x | Yes | 110ms | 44.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/WZ0i7RZW5BRpJu6/download)
FCOS_imprv_dcnv2_X_101_32x8d_FPN_2x | Yes | 143ms | 46.4 | [download](https://cloudstor.aarnet.edu.au/plus/s/08UK0OP67TogLCU/download)
FCOS_imprv_X_101_64x4d_FPN_2x | Yes | 112ms | 44.7 | [download](https://cloudstor.aarnet.edu.au/plus/s/rKOJtwvJwcKVOz8/download)
FCOS_imprv_dcnv2_X_101_64x4d_FPN_2x | Yes | 144ms | 46.6 | [download](https://cloudstor.aarnet.edu.au/plus/s/jdtVmG7MlugEXB7/download)

*Note that `imprv` denotes `improvements` in our paper Table 3. These almost cost-free changes improve the performance by ~1.5% in total. Thus, we highly recommend to use them. The following are the original models presented in our initial paper.*

Model | Multi-scale training | Testing time / im | AP (minival) | AP (test-dev) | Link
--- |:---:|:---:|:---:|:---:|:---:
FCOS_R_50_FPN_1x | No | 45ms | 37.1 | 37.4 | [download](https://cloudstor.aarnet.edu.au/plus/s/dDeDPBLEAt19Xrl/download)
FCOS_R_101_FPN_2x | Yes | 59ms | 41.4 | 41.5 | [download](https://cloudstor.aarnet.edu.au/plus/s/vjL3L0AW7vnhRTo/download)
FCOS_X_101_32x8d_FPN_2x | Yes | 110ms | 42.5 | 42.7 | [download](https://cloudstor.aarnet.edu.au/plus/s/U5myBfGF7MviZ97/download)
FCOS_X_101_64x4d_FPN_2x | Yes | 113ms | 43.0 | 43.2 | [download](https://cloudstor.aarnet.edu.au/plus/s/wpwoCi4S8iajFi9/download)

**MobileNets:**

*We update batch normalization for MobileNet based models. If you want to use SyncBN, please install pytorch 1.1 or later.*

Model | Training batch size | Multi-scale training | Testing time / im | AP (minival) | Link
--- |:---:|:---:|:---:|:---:|:---:
FCOS_syncbn_bs32_c128_MNV2_FPN_1x | 32 | No | 26ms | 30.9 | [download](https://cloudstor.aarnet.edu.au/plus/s/3GKwaxZhDSOlCZ0/download)
FCOS_syncbn_bs32_MNV2_FPN_1x | 32 | No | 33ms | 33.1 | [download](https://cloudstor.aarnet.edu.au/plus/s/OpJtCJLj104i2Yc/download)
FCOS_bn_bs16_MNV2_FPN_1x | 16 | No | 44ms | 31.0 | [download](https://cloudstor.aarnet.edu.au/plus/s/B6BrLAiAEAYQkcy/download)

[1] *1x and 2x mean the model is trained for 90K and 180K iterations, respectively.* \
[2] *All results are obtained with a single model and without any test time data augmentation such as multi-scale, flipping and etc..* \
[3] *`c128` denotes the model has 128 (instead of 256) channels in towers (i.e., `MODEL.RESNETS.BACKBONE_OUT_CHANNELS` in [config](https://github.com/tianzhi0549/FCOS/blob/master/configs/fcos/fcos_syncbn_bs32_c128_MNV2_FPN_1x.yaml#L10)).* \
[4] *`dcnv2` denotes deformable convolutional networks v2. Note that for ResNet based models, we apply deformable convolutions from stage c3 to c5 in backbones. For ResNeXt based models, only stage c4 and c5 use deformable convolutions. All models use deformable convolutions in the last layer of detector towers.* \
[5] *The model `FCOS_imprv_dcnv2_X_101_64x4d_FPN_2x` with multi-scale testing achieves 49.0% in AP on COCO test-dev. Please use `TEST.BBOX_AUG.ENABLED True` to enable multi-scale testing.*

## Training

The following command line will train FCOS_imprv_R_50_FPN_1x on 8 GPUs with Synchronous Stochastic Gradient Descent (SGD):

    python -m torch.distributed.launch \
        --nproc_per_node=8 \
        --master_port=$((RANDOM + 10000)) \
        tools/train_net.py \
        --config-file configs/fcos/fcos_imprv_R_50_FPN_1x.yaml \
        DATALOADER.NUM_WORKERS 2 \
        OUTPUT_DIR training_dir/fcos_imprv_R_50_FPN_1x
        
Note that:
1) If you want to use fewer GPUs, please change `--nproc_per_node` to the number of GPUs. No other settings need to be changed. The total batch size does not depends on `nproc_per_node`. If you want to change the total batch size, please change `SOLVER.IMS_PER_BATCH` in [configs/fcos/fcos_R_50_FPN_1x.yaml](configs/fcos/fcos_R_50_FPN_1x.yaml).
2) The models will be saved into `OUTPUT_DIR`.
3) If you want to train FCOS with other backbones, please change `--config-file`.
4) If you want to train FCOS on your own dataset, please follow this instruction [#54](https://github.com/tianzhi0549/FCOS/issues/54#issuecomment-497558687).
5) Now, training with 8 GPUs and 4 GPUs can have the same performance. Previous performance gap was because we did not synchronize `num_pos` between GPUs when computing loss. 

## Contributing to the project
Any pull requests or issues are welcome.

## Citations
Please consider citing our paper in your publications if the project helps your research. BibTeX reference is as follows.
```
@inproceedings{tian2019fcos,
  title   =  {{FCOS}: Fully Convolutional One-Stage Object Detection},
  author  =  {Tian, Zhi and Shen, Chunhua and Chen, Hao and He, Tong},
  booktitle =  {Proc. Int. Conf. Computer Vision (ICCV)},
  year    =  {2019}
}
```

# Acknowledgments
We would like to thank [@yqyao](https://github.com/yqyao) for the tricks of center sampling and GIoU.  We also thank [@bearcatt](https://github.com/bearcatt) for his suggestion of positioning the center-ness branch with box regression (refer to [#89](https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042)).    

## License

For academic use, this project is licensed under the 2-clause BSD License - see the LICENSE file for details. For commercial use, please contact the authors. 

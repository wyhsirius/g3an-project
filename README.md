# G3AN: Disentangling Appearance and Motion for Video Generation
### [Project Page](https://wyhsirius.github.io/G3AN/) | [Paper](https://arxiv.org/pdf/1912.05523.pdf)
This is the official PyTorch implementation of the CVPR 2020 paper "G3AN: Disentangling Appearance and Motion for Video Generation"

<img src="demo.gif" width="500">

## Requirements
- Python 3.6
- cuda 9.2
- cudnn 7.1
- PyTorch 1.4+
- scikit-video
- tensorboard
- moviepy

## Dataset
You can download the original UvA-NEMO datest from https://www.uva-nemo.org/ and use https://github.com/1adrianb/face-alignment to crop face regions. We also provide our preprocessed version [here](https://drive.google.com/file/d/1aB7w3d1Ev9Iniui1LTuhLi7zNVK-Wxen/view?usp=sharing).

## Pretrained model
Download the G3AN pretrained model on UvA-NEMO from [here](https://drive.google.com/file/d/1sDkWELQHsQqg0MUR-DJsM3YpSyenTX-S/view?usp=sharing).

## Inference
1. For sampling NUM videos and saving them under ./demos/EXP_NAME

```shell script
python demo_random.py --model_path $MODEL_PATH --n $NUM --demo_name $EXP_NAME
```

2. For sampling N appearances with M motions and saving them under ./demos/EXP_NAME
```shell script
python demo_nxm.py --model_path $MODEL_PATH --n_za_test $N --n_zm_test $M --demo_name $EXP_NAME
```

3. For sampling N appearances with different video lengthes (9 different video lengthes) and saving them under ./demos/EXP_NAME
```shell script
python demo_multilength.py --model_path $MODEL_PATH --n_za_test $N --demo_name $EXP_NAME
```

## Training
```shell script
python train.py --data_path $DATASET --exp_name $EXP_NAME
```

## Evaluation
1. Generate 5000 videos for evaluation, save them in $GEN_PATH
```shell script
python generate_videos.py --gen_path $GEN_PATH
```

2. Move into evaluation folder
```shell script
cd evaluation
``` 
Download feature extractor **resnext-101-kinetics.pth** from [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M) to the current folder. Pre-computed UvA_NEMO dataset stats can be found in stats/uva.npz. If you would like to compute it youeself, save all the training videos in $UVA_PATH and run
```shell script
python precalc_stats.py --data_path $UVA_PATH
```
To compute FID
```shell script
python fid.py $GEN_PATH stats/uva.npz
```
You can obtain FID around 80 ~ 83 (better than reported number on the paper) by evaluating provided model. Here I improve the original video discriminator by using a (2+1)D ConvNets instead of 3D ConvNets.

## TODOs
- [x] Unconditional Generation
- [x] Evaluation
- [ ] Conditional Generation

## Citation
If you find this code useful for your research, please consider citing our paper:
```
@InProceedings{Wang_2020_CVPR,
    author = {Wang, Yaohui and Bilinski, Piotr and Bremond, Francois and Dantcheva, Antitza},
    title = {G3AN: Disentangling Appearance and Motion for Video Generation},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

## Acknowledgement
Part of the evaluation code is adapted from [evan](https://github.com/raahii/evan). I moved most of the operations from CPU into GPU to accelerate the computation. We thank the authors for their inspiration and contribution to the community.

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
- tensoboard
- moviepy

## Dataset
You can download the original UvA-NEMO datest from https://www.uva-nemo.org/ and use https://github.com/1adrianb/face-alignment to crop face regions. We also provide our preprocessed version [here](https://filesender.renater.fr/download.php?token=53549086-caa6-4178-af12-ec10049570c3&files_ids=2070047).

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

## TODOs
- [x] Unconditional Generation
- [ ] Conditional Generation

## Citation
If you find this code useful for your research, please consider citing our paper:
```bibtex
@InProceedings{Wang_2020_CVPR,
    author = {Wang, Yaohui and Bilinski, Piotr and Bremond, Francois and Dantcheva, Antitza},
    title = {G3AN: Disentangling Appearance and Motion for Video Generation},
    booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2020}
}
```

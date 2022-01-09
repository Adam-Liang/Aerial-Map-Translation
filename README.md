# Translation of Aerial Image Into Digital Map via Discriminative Segmentation and Creative Generation
This is the code for Translation of Aerial Image Into Digital Map via Discriminative Segmentation and Creative Generation (TGRS) 
[Conference Paper Link](https://ieeexplore.ieee.org/document/9540226) 

## Dataset


## Code

Our dataset AIDOMG can be donwloaded from [here](https://pan.baidu.com/s/1HsludHYqlHtjTPPenP-W2w) with code "d8wf".

### Prerequisites

- Python 3.8
- PyTorch 1.7 with GPU
- opencv-python
- scikit-image
- tensorboard

### Train and Test

1. Download pretrain model for backbone from [here](https://pan.baidu.com/s/1CSMPyyfcyISI3zsAr0guvA) with code "cag8" and put it under ./src/pix2pixHD;
2. Create a training dataset folder according to your needs: The following subfolders are required under the folder: 'testA','testB','test_seg','trainA','trainB','train_seg'; it is recommended to use the default training/testing set division;
3. Modify '--dataroot' in ./shs/train_haikou.sh to your data set path and run it.

## Citing

If you use any part of our research, please consider citing:

```bibtex
@article{fu2021translation,
  title={Translation of Aerial Image Into Digital Map via Discriminative Segmentation and Creative Generation},
  author={Fu, Ying and Liang, Shuaizhe and Chen, Dongdong and Chen, Zhanlong},
  journal={IEEE Transactions on Geoscience and Remote Sensing},
  year={2021},
  publisher={IEEE}
}
```


## Acknowledgement
Our work and implementations are inspired by following projects:
[ESTRNN](https://github.com/phillipi/pix2pix)
[pix2pixHD](https://github.com/NVIDIA/pix2pixHD)
[DeepLabv3+](https://github.com/tensorflow/models/tree/master/research/deeplab)

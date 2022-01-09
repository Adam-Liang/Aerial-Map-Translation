'''计算p2pHD的参数量'''
'''由train_img2map.py拷贝而来并添加参数量计算内容'''

import os
from os import path as osp
import sys

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
sys.path.append(osp.join(sys.path[0], '../'))
sys.path.append(osp.join(sys.path[0], '../../'))
# sys.path.append(osp.join(sys.path[0], '../../../'))
import time
import torch
import torch.nn as nn
from src.utils.train_utils import model_accelerate, get_device, mean, get_lr
from src.pix2pixHD.train_config import config
from src.pix2pixHD.networks import get_G, get_D, get_E
from torch.optim import Adam
from src.pix2pixHD.hinge_lr_scheduler import get_hinge_scheduler
from src.utils.logger import ModelSaver, Logger
from src.datasets import get_pix2pix_maps_dataloader
from src.pix2pixHD.utils import get_edges, label_to_one_hot, get_encode_features
from src.utils.visualizer import Visualizer
from tqdm import tqdm
from torchvision import transforms
from src.pix2pixHD.criterion import get_GANLoss, get_VGGLoss, get_DFLoss, get_low_level_loss
from tensorboardX import SummaryWriter
from src.pix2pixHD.utils import from_std_tensor_save_image, create_dir
from pix2pixHD.eval_img2map import eval, get_fid
from src.data.image_folder import make_dataset
import shutil
import numpy as np
from PIL import Image

# from src.pix2pixHD.deeplabv3.model.deeplabv3 import DeepLabV3
from src.pix2pixHD.deeplabv3plus.deeplabv3plus import Configuration
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.deeplabv3plus_seg import deeplabv3plus
import torch.nn.functional as F




def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    logger = Logger(save_path=args.save, json_name='img2map')
    epoch_now = len(logger.get_data('G_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'E', 'G_optimizer', 'D_optimizer', 'E_optimizer',
                                        'G_scheduler', 'D_scheduler', 'E_scheduler','DLV3P',"DLV3P_global_optimizer",
                                        "DLV3P_backbone_optimizer","DLV3P_global_scheduler","DLV3P_backbone_scheduler"])
    visualizer = Visualizer(keys=['image', 'encode_feature', 'fake', 'label', 'instance'])
    sw = SummaryWriter(args.tensorboard_path)
    G = get_G(args)
    D = get_D(args)
    model_saver.load('G', G)
    model_saver.load('D', D)

    params_G=sum([param.nelement() for param in G.parameters()])
    params_D = sum([param.nelement() for param in D.parameters()])
    print(f"{params_G}, {params_D}")
    print(f"{params_G}")
    sys.exit(0) # 测完退出


if __name__ == '__main__':
    args = config()
    assert args.feat_num == 0
    assert args.use_instance == 0

    from src.pix2pixHD.myutils import seed_torch

    print(f'set seed as {args.seed}!')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass

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
from evaluation.fid.fid_score import fid_score

import torch.nn.functional as F


def eval(args, model, data_loader,model_seg=None):
    device = get_device(args)
    data_loader = tqdm(data_loader)
    model.eval()
    model = model.to(device)
    if not(model_seg is None):
        model_seg.eval()
        model_seg=model_seg.to(device)
        seg_dir=osp.join(args.save, 'seg_result')
        label_preds = []
        label_targets = []
    fake_dir = osp.join(args.save, 'fake_result')
    real_dir = osp.join(args.save, 'real_result')
    A_dir = osp.join(args.save, 'real_source')
    seg_dir = osp.join(args.save, 'seg_result')
    create_dir(real_dir)
    create_dir(fake_dir)
    create_dir(A_dir)
    create_dir(seg_dir)

    for i, sample in enumerate(data_loader):
        # imgs = sample['image'].to(device)
        # maps = sample['map'].to(device)
        # im_name = sample['im_name']
        imgs = sample['A'].to(device)
        maps = sample['B'].to(device)
        im_name = sample['A_paths']
        with torch.no_grad():
            if model_seg is None:
                fakes = model(imgs)
            else:
                outputs, feature_map = model_seg(imgs)
                input_2 = F.upsample(feature_map, size=(64, 64), mode="bilinear")  # BS*256*64*64
                input_3 = F.upsample(feature_map, size=(128, 128), mode="bilinear")  # BS*256*128*128
                fakes = model(imgs, input_2, input_3)
                # 以下为计算iou的准备
                bs,n_class,h,w=outputs.shape
                outs=outputs.data.cpu().numpy()
                pred=outs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
                target = sample['seg'].cpu().numpy().reshape(bs, h, w)
                label_preds.append(pred)
                label_targets.append(target)

        batch_size = imgs.size(0)
        if not (model_seg is None):
            from src.pix2pixHD.myutils import pred2gray
            outputs=pred2gray(outputs)
        for b in range(batch_size):
            file_name = osp.split(im_name[b])[0].split(os.sep)[-2]+'_'+osp.split(im_name[b])[0].split(os.sep)[-1]+'_'+osp.split(im_name[b])[-1].split('.')[0]
            real_file = osp.join(real_dir, f'{file_name}.tif')
            fake_file = osp.join(fake_dir, f'{file_name}.tif')
            A_file = osp.join(A_dir, f'{file_name}.tif')
            if not (model_seg is None):
                seg_file = osp.join(seg_dir, f'{file_name}.tif')
            # if not(model_seg is None):
            #     seg_file = osp.join(seg_dir, f'{file_name}.tif')
                # from_std_tensor_save_image(filename=seg_file, data=torch.unsqueeze(outputs[b],0).cpu())

            from_std_tensor_save_image(filename=real_file, data=maps[b].cpu())
            from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
            from_std_tensor_save_image(filename=A_file, data=imgs[b].cpu())
            if not (model_seg is None):
                from_std_tensor_save_image(filename=seg_file, data=outputs[b].cpu())
        pass
    pass
    fid = fid_score(real_path=real_dir, fake_path=fake_dir, gpu=str(args.gpu))
    print(f'===> fid score:{fid:.4f}')

    iou=None
    if not(model_seg is None):
        from src.pix2pixHD.eval_iou import label_accuracy_score
        _,_,iou,_,_=label_accuracy_score(label_targets, label_preds, n_class)

    model.train()
    return fid,iou


def get_fid(args):
    fake_dir = osp.join(args.save, 'fake_result')
    real_dir = osp.join(args.save, 'real_result')
    fid = fid_score(real_path=real_dir, fake_path=fake_dir, gpu=str(args.gpu))
    print(f'===> fid score:{fid:.4f}')
    return fid


if __name__ == '__main__':
    args = config()
    get_fid(args)
    try:
        get_fid(args)
    except Exception:
        assert args.feat_num == 0
        assert args.use_instance == 0
        model_saver = ModelSaver(save_path=args.save,
                                 name_list=['G', 'D', 'E', 'G_optimizer', 'D_optimizer', 'E_optimizer',
                                            'G_scheduler', 'D_scheduler', 'E_scheduler'])
        visualizer = Visualizer(keys=['image', 'encode_feature', 'fake', 'label', 'instance'])
        sw = SummaryWriter(args.tensorboard_path)
        G = get_G(args)
        model_saver.load('G', G)
        eval(args, G, get_pix2pix_maps_dataloader(args, train=False))
        pass

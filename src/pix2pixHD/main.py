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
from src.data.image_folder import make_dataset
import shutil
import numpy as np
from PIL import Image

from src.pix2pixHD.deeplabv3plus.deeplabv3plus import Configuration
from src.pix2pixHD.deeplabv3plus.deeplabv3plus.my_deeplabv3plus_featuremap import deeplabv3plus
import torch.nn.functional as F

from src.pix2pixHD.deeplabv3plus.lovasz_losses import lovasz_softmax

from util.util import tensor2im  # 注意，该函数使用0.5与255恢复可视图像，所以如果是ImageNet标准化的可能会有色差？这里都显示试一下
from src.pix2pixHD.myutils import pred2gray, gray2rgb

from src.pix2pixHD.deeplabv3plus.focal_loss import FocalLoss
from evaluation.fid.fid_score import fid_score
import json

def eval_fidiou(args, model_G,model_seg, data_loader):
    device = get_device(args)
    data_loader = tqdm(data_loader)
    model_G.eval()
    model_seg.eval()
    model_G = model_G.to(device)
    model_seg = model_seg.to(device)

    label_preds = []
    label_targets = []

    real_seg_dir = osp.join(args.save, 'real_seg')
    real_dir = osp.join(args.save, 'real_result')
    A_dir = osp.join(args.save, 'real_source')
    seg_dir = osp.join(args.save, 'seg_result')
    fake_dir=osp.join(args.save, 'fake_result')
    create_dir(real_dir)
    create_dir(real_seg_dir)
    create_dir(A_dir)
    create_dir(seg_dir)
    create_dir(fake_dir)

    for i, sample in enumerate(data_loader):
        inputs, labels = sample['A_seg'], sample['seg'].squeeze(dim=1)
        inputs = inputs.cuda() if args.gpu else inputs
        labels = labels.cuda() if args.gpu else labels
        imgs = sample['A'].to(device)
        maps = sample['B'].to(device)

        outputs, feature_map = model_seg(inputs)
        bs, n_class, h, w = outputs.shape
        outs = outputs.data.cpu().numpy()
        pred = outs.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
        target = labels.cpu().numpy().reshape(bs, h, w)
        label_preds.append(pred)
        label_targets.append(target)

        # seg_ret = pred2gray(outputs).unsqueeze(1).type(torch.FloatTensor).to(device)  # bs*1*h*w
        feature_map = feature_map.detach()
        imgs_plus=torch.cat((imgs,feature_map),1)
        fakes = model_G(imgs_plus).detach()

        batch_size = inputs.size(0)
        im_name = sample['A_paths']
        for b in range(batch_size):
            file_name = osp.split(im_name[b])[0].split(os.sep)[-2]+'_'+osp.split(im_name[b])[0].split(os.sep)[-1]+'_'+osp.split(im_name[b])[-1].split('.')[0]
            real_file = osp.join(real_dir, f'{file_name}.tif')
            real_seg_file = osp.join(real_seg_dir, f'{file_name}.tif')
            A_file = osp.join(A_dir, f'{file_name}.tif')
            seg_file = osp.join(seg_dir, f'{file_name}.tif')
            fake_file=osp.join(fake_dir, f'{file_name}.tif')

            from_std_tensor_save_image(filename=real_file, data=sample['B'][b].cpu())
            from_std_tensor_save_image(filename=A_file, data=sample['A'][b].cpu())
            from_std_tensor_save_image(filename=fake_file, data=fakes[b].cpu())
            tmpimg= sample['seg'][b].data.cpu().numpy()
            tmpimg = gray2rgb(tmpimg)
            tmpimg=Image.fromarray(tmpimg)
            tmpimg.save(fp=real_seg_file)

            tmpimg = gray2rgb(pred[b])
            tmpimg = Image.fromarray(tmpimg)
            tmpimg.save(fp=seg_file)

    fid = fid_score(real_path=real_dir, fake_path=fake_dir, gpu=str(args.gpu))
    print(f'===> fid score:{fid:.4f}')
    iou=None
    from src.pix2pixHD.eval_iou import label_accuracy_score
    _,_,iou,_,_=label_accuracy_score(label_targets, label_preds, n_class)
    print(f'===> iou score:{iou:.4f}')

    model_seg.train()
    model_G.train()
    return fid,iou

def label_nums(data_loader,label_num=5): # 遍历dataloader，计算其所有图像中分割GT各label的pix总数
    ret=[]
    for i in range(label_num):
        ret.append(0)
    for step, sample in enumerate(data_loader):
        seg=sample["seg"]
        for i in range(label_num):
            ret[i]+=(seg==i).sum().item()
    return ret




def train(args, get_dataloader_func=get_pix2pix_maps_dataloader):
    with open(os.path.join(args.save,'args.json'), 'w') as f:
        json.dump(vars(args), f)
    logger = Logger(save_path=args.save, json_name='img2map_seg')
    epoch_now = len(logger.get_data('FOCAL_loss'))

    model_saver = ModelSaver(save_path=args.save,
                             name_list=['G', 'D', 'G_optimizer', 'D_optimizer',
                                        'G_scheduler', 'D_scheduler','DLV3P',"DLV3P_global_optimizer",
                                        "DLV3P_backbone_optimizer","DLV3P_global_scheduler","DLV3P_backbone_scheduler",
                                        'best_G', 'best_D', 'best_G_optimizer', 'best_D_optimizer',
                                        'best_G_scheduler', 'best_D_scheduler', 'best_DLV3P',"best_DLV3P_global_optimizer",
                                        "best_DLV3P_backbone_optimizer", "best_DLV3P_global_scheduler","best_DLV3P_backbone_scheduler"
                                        ])

    sw = SummaryWriter(args.tensorboard_path)

    G = get_G(args,input_nc=3+256) # 3+256，256为分割网络输出featuremap的通道数
    D = get_D(args)
    model_saver.load('G', G)
    model_saver.load('D', D)

    cfg=Configuration()
    cfg.MODEL_NUM_CLASSES=args.label_nc
    DLV3P=deeplabv3plus(cfg)
    if args.gpu:
        # DLV3P=nn.DataParallel(DLV3P)
        DLV3P=DLV3P.cuda()
    model_saver.load('DLV3P', DLV3P)

    G_optimizer = Adam(G.parameters(), lr=args.G_lr, betas=(args.beta1, 0.999))
    D_optimizer = Adam(D.parameters(), lr=args.D_lr, betas=(args.beta1, 0.999))

    seg_global_params, seg_backbone_params=DLV3P.get_paras()
    DLV3P_global_optimizer = torch.optim.Adam([{'params': seg_global_params, 'initial_lr': args.seg_lr_global}], lr=args.seg_lr_global,betas=(args.beta1, 0.999))
    DLV3P_backbone_optimizer = torch.optim.Adam([{'params': seg_backbone_params, 'initial_lr': args.seg_lr_backbone}], lr=args.seg_lr_backbone, betas=(args.beta1, 0.999))

    model_saver.load('G_optimizer', G_optimizer)
    model_saver.load('D_optimizer', D_optimizer)
    model_saver.load('DLV3P_global_optimizer', DLV3P_global_optimizer)
    model_saver.load('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

    G_scheduler = get_hinge_scheduler(args, G_optimizer)
    D_scheduler = get_hinge_scheduler(args, D_optimizer)
    DLV3P_global_scheduler=torch.optim.lr_scheduler.LambdaLR(DLV3P_global_optimizer, lr_lambda=lambda epoch:(1 - epoch/args.epochs)**0.9,last_epoch=epoch_now)
    DLV3P_backbone_scheduler = torch.optim.lr_scheduler.LambdaLR(DLV3P_backbone_optimizer,lr_lambda=lambda epoch: (1 - epoch / args.epochs) ** 0.9,last_epoch=epoch_now)

    model_saver.load('G_scheduler', G_scheduler)
    model_saver.load('D_scheduler', D_scheduler)
    model_saver.load('DLV3P_global_scheduler', DLV3P_global_scheduler)
    model_saver.load('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)

    D_scheduler.step(epoch_now) # 调整lr便于finetrain
    G_scheduler.step(epoch_now)

    device = get_device(args)

    GANLoss = get_GANLoss(args)
    if args.use_ganFeat_loss:
        DFLoss = get_DFLoss(args)
    if args.use_vgg_loss:
        VGGLoss = get_VGGLoss(args)
    if args.use_low_level_loss:
        LLLoss = get_low_level_loss(args)

    # CE_loss=nn.CrossEntropyLoss(ignore_index=255)
    LVS_loss = lovasz_softmax
    data_loader_focal = get_dataloader_func(args, train=True)
    data_loader_focal = tqdm(data_loader_focal)
    alpha = label_nums(data_loader_focal,label_num=args.label_nc)
    # alpha = [1,1,1,1,1]
    tmp_min = min(alpha)
    assert tmp_min > 0
    for i in range(len(alpha)):
        alpha[i] = tmp_min / alpha[i]
    if args.focal_alpha_revise:
        assert len(args.focal_alpha_revise) == len(alpha)
        for i in range(len(alpha)):
            alpha[i]=alpha[i]*args.focal_alpha_revise[i]
    print(alpha)
    FOCAL_loss = FocalLoss(gamma=2, alpha=alpha)


    if epoch_now==args.epochs:
        print('get final models')
        iou = eval_fidiou(args,model_G=G, model_seg=DLV3P,data_loader=get_pix2pix_maps_dataloader(args, train=False))
        logger.log(key='iou', data=iou)
        # if iou < logger.get_max(key='FID'):
        #     model_saver.save(f'DLV3P_{iou:.4f}', DLV3P)
        sw.add_scalar('eval/iou', iou, epoch_now)

    for epoch in range(epoch_now, args.epochs):
        G_loss_list = []
        D_loss_list = []
        # CE_loss_list = []
        LVS_loss_list=[]
        FOCAL_loss_list=[]

        data_loader = get_dataloader_func(args, train=True,flag=(2 if args._usefakelen else 0)) # flag=2:使用虚假的数据长度
        data_loader = tqdm(data_loader)

        for step, sample in enumerate(data_loader):
            # 先训练deeplabv3+
            imgs_seg = sample['A_seg'].to(device)  # (shape: (batch_size, 3, img_h, img_w))
            label_imgs = sample['seg'].type(torch.LongTensor).to(device)  # (shape: (batch_size, img_h, img_w))
            # print(label_imgs)
            # print(label_imgs.max())
            # print(label_imgs.min())

            # imgs_show=sample['A'].to(device) # (shape: (batch_size, 3, img_h, img_w))
            # maps_show= sample['B'].to(device)  # (shape: (batch_size, 3, img_h, img_w))

            outputs,feature_map = DLV3P(imgs_seg)  # (shape: (batch_size, num_classes, img_h, img_w))
            # feature_map=feature_map.detach()

            # compute the loss:
            # ce_loss = CE_loss(outputs, label_imgs)
            # ce_loss_value = ce_loss.data.cpu().numpy()
            soft_outputs = torch.nn.functional.softmax(outputs, dim=1)
            lvs_loss = LVS_loss(soft_outputs, label_imgs, ignore=255)
            lvs_loss_value = lvs_loss.data.cpu().numpy()
            focal_loss=FOCAL_loss(outputs,label_imgs)
            focal_loss_value=focal_loss.data.cpu().numpy()

            seg_loss = (focal_loss+lvs_loss)*0.5

            # optimization step:
            # DLV3P_global_optimizer.zero_grad()  # (reset gradients)
            # DLV3P_backbone_optimizer.zero_grad()
            # seg_loss.backward()  # (compute gradients)
            # DLV3P_global_optimizer.step()  # (perform optimization step)
            # DLV3P_backbone_optimizer.step()

            # 然后训练GAN
            imgs = sample['A'].to(device)
            maps = sample['B'].to(device)
            # feature_map=feature_map.detach()
            imgs_plus=torch.cat((imgs,feature_map),1) # bs*(3+256)*h*w

            # train the Discriminator
            D_optimizer.zero_grad()
            reals_maps = torch.cat([imgs.float(), maps.float()], dim=1)

            fakes = G(imgs_plus).detach()
            fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)

            D_real_outs = D(reals_maps)
            D_real_loss = GANLoss(D_real_outs, True)

            D_fake_outs = D(fakes_maps)
            D_fake_loss = GANLoss(D_fake_outs, False)

            D_loss = 0.5 * (D_real_loss + D_fake_loss)
            D_loss = D_loss.mean()
            D_loss.backward()
            D_loss = D_loss.item()
            D_optimizer.step()

            # train generator and encoder
            # G_optimizer.zero_grad()
            fakes = G(imgs_plus)
            fakes_maps = torch.cat([imgs.float(), fakes.float()], dim=1)
            D_fake_outs = D(fakes_maps)

            gan_loss = GANLoss(D_fake_outs, True)

            G_loss = 0
            G_loss += gan_loss
            gan_loss = gan_loss.mean().item()

            if args.use_vgg_loss:
                vgg_loss = VGGLoss(fakes, imgs)
                G_loss += args.lambda_feat * vgg_loss
                vgg_loss = vgg_loss.mean().item()
            else:
                vgg_loss = 0.

            if args.use_ganFeat_loss:
                df_loss = DFLoss(D_fake_outs, D_real_outs)
                G_loss += args.lambda_feat * df_loss
                df_loss = df_loss.mean().item()
            else:
                df_loss = 0.

            if args.use_low_level_loss:
                ll_loss = LLLoss(fakes, maps)
                G_loss += args.lambda_feat * ll_loss
                ll_loss = ll_loss.mean().item()
            else:
                ll_loss = 0.

            G_loss = G_loss.mean()
            G_seg_loss =args._1002arg_GANloss_alpha*G_loss+args._1002arg_segloss_alpha*seg_loss
            G_loss = G_loss.item()
            seg_loss=seg_loss.item()

            G_optimizer.zero_grad()
            DLV3P_global_optimizer.zero_grad()  # (reset gradients)
            DLV3P_backbone_optimizer.zero_grad()
            G_seg_loss.backward()
            G_optimizer.step()
            DLV3P_global_optimizer.step()  # (perform optimization step)
            DLV3P_backbone_optimizer.step()

            data_loader.write(f'Epochs:{epoch}  | Dloss:{D_loss:.6f} | Gloss:{G_loss:.6f}'
                              f'| GANloss:{gan_loss:.6f} | VGGloss:{vgg_loss:.6f} | DFloss:{df_loss:.6f} '
                              f'| LLloss:{ll_loss:.6f} | lr_gan:{get_lr(G_optimizer):.8f}'
                              f'| FOCAL_loss:{focal_loss_value:.6f}|LVS_loss:{lvs_loss_value:.6f} '
                              f'| lr_global:{get_lr(DLV3P_global_optimizer):.8f}| lr_backbone:{get_lr(DLV3P_backbone_optimizer):.8f}')

            G_loss_list.append(G_loss)
            D_loss_list.append(D_loss)
            # CE_loss_list.append(ce_loss_value)
            LVS_loss_list.append(lvs_loss_value)
            FOCAL_loss_list.append(focal_loss_value)

            # tensorboard log
            if args.tensorboard_log and step % args.tensorboard_log == 0:  # defalut is 5
                total_steps = epoch * len(data_loader) + step
                sw.add_scalar('Loss1/G', G_loss, total_steps)
                sw.add_scalar('Loss1/seg', seg_loss, total_steps)
                sw.add_scalar('Loss1/G_seg', G_seg_loss, total_steps)
                sw.add_scalar('Loss/G', G_loss, total_steps)
                sw.add_scalar('Loss/D', D_loss, total_steps)
                sw.add_scalar('Loss/gan', gan_loss, total_steps)
                sw.add_scalar('Loss/vgg', vgg_loss, total_steps)
                sw.add_scalar('Loss/df', df_loss, total_steps)
                sw.add_scalar('Loss/ll', ll_loss, total_steps)
                # sw.add_scalar('Loss/CE', ce_loss_value, total_steps)
                sw.add_scalar('Loss/LVS', lvs_loss_value, total_steps)
                sw.add_scalar('Loss/focal', focal_loss_value, total_steps)
                sw.add_scalar('LR/G', get_lr(G_optimizer), total_steps)
                sw.add_scalar('LR/D', get_lr(D_optimizer), total_steps)
                sw.add_scalar('LR/global_seg', get_lr(DLV3P_global_optimizer), total_steps)
                sw.add_scalar('LR/backbone_seg', get_lr(DLV3P_backbone_optimizer), total_steps)

                sw.add_image('img2/realA', tensor2im(imgs.data), total_steps, dataformats='HWC')
                sw.add_image('img2/fakeB', tensor2im(fakes.data), total_steps, dataformats='HWC')
                sw.add_image('img2/realB', tensor2im(maps.data), total_steps, dataformats='HWC')
                tmpsegmap = pred2gray(outputs)
                tmpsegmap = tmpsegmap[0].data.numpy()
                tmpsegmap = gray2rgb(tmpsegmap)
                sw.add_image('img2/fake_segB', tmpsegmap, total_steps, dataformats='HWC')
                tmpsegmap = label_imgs[0].data.cpu().numpy()
                tmpsegmap = gray2rgb(tmpsegmap)
                sw.add_image('img2/real_segB', tmpsegmap, total_steps, dataformats='HWC')

        D_scheduler.step(epoch)
        G_scheduler.step(epoch)
        DLV3P_global_scheduler.step()
        DLV3P_backbone_scheduler.step()
        if epoch % args._val_frequency == 0 or epoch == (args.epochs-1):
            import copy
            args2=copy.deepcopy(args)
            args2.batch_size=args.batch_size_eval
            fid,iou = eval_fidiou(args,model_G=G, model_seg=DLV3P,data_loader=get_pix2pix_maps_dataloader(args2, train=False))
            if fid < logger.get_min(key='FID') == 0 :
            # if epoch >= 80 :
                # model_saver.save('best_G', G)
                # model_saver.save('best_D', D)
                # model_saver.save('best_DLV3P', DLV3P)
                # model_saver.save('best_G_optimizer', G_optimizer)
                # model_saver.save('best_D_optimizer', D_optimizer)
                # model_saver.save('best_DLV3P_global_optimizer', DLV3P_global_optimizer)
                # model_saver.save('best_DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)
                # model_saver.save('best_G_scheduler', G_scheduler)
                # model_saver.save('best_D_scheduler', D_scheduler)
                # model_saver.save('best_DLV3P_global_scheduler', DLV3P_global_scheduler)
                # model_saver.save('best_DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)
                model_saver.save(f'best_G_{epoch}', G)
                model_saver.save(f'best_D_{epoch}', D)
                model_saver.save(f'best_DLV3P_{epoch}', DLV3P)
                model_saver.save(f'best_G_optimizer_{epoch}', G_optimizer)
                model_saver.save(f'best_D_optimizer_{epoch}', D_optimizer)
                model_saver.save(f'best_DLV3P_global_optimizer_{epoch}', DLV3P_global_optimizer)
                model_saver.save(f'best_DLV3P_backbone_optimizer_{epoch}', DLV3P_backbone_optimizer)
                model_saver.save(f'best_G_scheduler_{epoch}', G_scheduler)
                model_saver.save(f'best_D_scheduler_{epoch}', D_scheduler)
                model_saver.save(f'best_DLV3P_global_scheduler_{epoch}', DLV3P_global_scheduler)
                model_saver.save(f'best_DLV3P_backbone_scheduler_{epoch}', DLV3P_backbone_scheduler)
            logger.log(key='FID', data=fid)
            logger.log(key='iou', data=iou)
            sw.add_scalar('eval/fid', fid, epoch)
            sw.add_scalar('eval/iou', iou, epoch)

        logger.log(key='D_loss', data=sum(D_loss_list) / float(len(D_loss_list)))
        logger.log(key='G_loss', data=sum(G_loss_list) / float(len(G_loss_list)))
        # logger.log(key='CE_loss', data=sum(CE_loss_list) / float(len(CE_loss_list)))
        logger.log(key='LVS_loss', data=sum(LVS_loss_list) / float(len(LVS_loss_list)))
        logger.log(key='FOCAL_loss', data=sum(FOCAL_loss_list) / float(len(FOCAL_loss_list)))
        logger.save_log()
        # logger.visualize()

        model_saver.save('G', G)
        model_saver.save('D', D)
        model_saver.save('DLV3P', DLV3P)

        model_saver.save('G_optimizer', G_optimizer)
        model_saver.save('D_optimizer', D_optimizer)
        model_saver.save('DLV3P_global_optimizer', DLV3P_global_optimizer)
        model_saver.save('DLV3P_backbone_optimizer', DLV3P_backbone_optimizer)

        model_saver.save('G_scheduler', G_scheduler)
        model_saver.save('D_scheduler', D_scheduler)
        model_saver.save('DLV3P_global_scheduler', DLV3P_global_scheduler)
        model_saver.save('DLV3P_backbone_scheduler', DLV3P_backbone_scheduler)



if __name__ == '__main__':
    args = config()

    # args.label_nc = 5

    from src.pix2pixHD.myutils import seed_torch

    print(f'\nset seed as {args.seed}!\n')
    seed_torch(args.seed)

    train(args, get_dataloader_func=get_pix2pix_maps_dataloader)

pass

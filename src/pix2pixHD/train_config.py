# -*- coding: utf-8 -*-
from __future__ import division

""" 
Trains a ResNeXt Model on Cifar10 and Cifar 100. Implementation as defined in:
Xie, S., Girshick, R., Dollár, P., Tu, Z., & He, K. (2016). 
Aggregated residual transformations for deep neural networks. 
arXiv preprint arXiv:1611.05431.

"""

import argparse
import os
import os.path as osp
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import time
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader


def create_dir(dir_path):
    if not osp.exists(dir_path):
        os.makedirs(dir_path)


def config():
    parser = argparse.ArgumentParser(description='Trains GAN on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # 后续试验新增控制参数

    parser.add_argument('--_0719arg_now_f1_road', type=float, default=0.0, help='now_f1_road.')
    parser.add_argument('--_0719arg_now_f1_building', type=float, default=0.0, help='now_f1_building.')

    parser.add_argument('--_0315arg_Dloss_alpha_forgeogan', type=float, default=1.0, help='Dloss_alpha.')
    parser.add_argument('--_0314arg_GANloss_alpha_forgeogan', type=float, default=1.0, help='GANloss_alpha.')

    parser.add_argument('--_val_frequency', type=int, default=10, help='how many epochs to eval.')

    parser.add_argument('--_1002arg_GANloss_alpha', type=float, default=1.0, help='GANloss_alpha.')
    parser.add_argument('--_1002arg_segloss_alpha', type=float, default=1.0, help='segloss_alpha.')

    parser.add_argument('--_0701arg_gradloss_L1_alpha', type=float, default=10, help='gardloss_alpha.')
    parser.add_argument('--_0701arg_gradloss_struct_alpha', type=float, default=1, help='gardloss_alpha.')
    parser.add_argument('--_usefakelen', type=float, default=1, help='_use fake lenth dataset.')

    # 确保复现用
    parser.add_argument('--seed', type=int, default=0, help='random seed, 0-65535')
    # 分割新增参数
    parser.add_argument('--seg_lr_global', type=float, default=0.0007, help='The Learning Rate if seg-model.')
    parser.add_argument('--seg_lr_backbone', type=float, default=0.00007, help='The Learning Rate if seg-model.')

    parser.add_argument('--no_flip', action='store_true', default=False, help='是否禁止随机翻转处理')
    parser.add_argument('--batch_size_eval', type=int, default=8, help='eval batch size') # 因为eval有额外内存占用，可设置该值小于训练的bs

    parser.add_argument('--focal_alpha_revise',nargs='*', type=float, default=None, help='revise factor of focal loss alpha')  # 手动为alpha设置乘性系数，默认为使所有label总体权重相同
    parser.add_argument('--a_loss', nargs='*', type=float, default=None,
                        help='revise factor of losses')  # 手动所用各个loss设置权重
    # Positional arguments
    parser.add_argument('--cifar_10_data_path', type=str, default='/Users/chenlinwei/dataset/cifar-10-batches-py',
                        help='Root for the Cifar dataset.')
    parser.add_argument('--mnist_data_path', type=str, default='/Users/chenlinwei/dataset',
                        help='Root for the MNIST dataset.')

    parser.add_argument('--voc2012_data_path', type=str, default='/Users/chenlinwei/dataset/VOCdevkit/VOC2012',
                        help='Root for the voc dataset.')
    parser.add_argument('--sbd_data_path', type=str, default='/Users/chenlinwei/dataset/SBD_FULL11355/dataset',
                        help='Root for the voc dataset.')

    parser.add_argument('--pix2pix_maps_data_path', type=str, default='/Users/chenlinwei/dataset/pix2pix_maps',
                        help='Root for the voc dataset.')

    parser.add_argument('--voc_repeat', type=int, default=1, help='Repeat for the voc dataset in train_voc_weakly.py')
    parser.add_argument('--sbd_repeat', type=int, default=1, help='Repeat for the voc dataset in train_voc_weakly.py')

    parser.add_argument('--dataset_repeat', type=int, default=1,
                        help='choose strong data size for semi superveised')

    parser.add_argument('--cityscapes_data_path', type=str, default='/Users/chenlinwei/dataset/cityscapes',
                        help='Root for the cityscape dataset.')
    parser.add_argument('--data_choose_size', type=int, default=None,
                        help='choose strong data size for semi superveised')

    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100'], help='Choose between Cifar10/100.')
    # Optimization options
    parser.add_argument('--optimizer', '-op', type=str, default='adam', help='Optimizer to train model.')
    parser.add_argument('--epochs', '-e', type=int, default=200, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0002, help='The Learning Rate.')
    parser.add_argument('--G_lr', type=float, default=0.0002, help='The Learning Rate if generator.')
    parser.add_argument('--D_lr', type=float, default=0.0002, help='The Learning Rate of discriminator.')
    parser.add_argument('--E_lr', type=float, default=0.0002, help='The Learning Rate of discriminator.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_batch_size', type=int, default=8)
    parser.add_argument('--scheduler', type=str, default='multi_step')
    parser.add_argument('--milestones', type=int, nargs='+', default=[25, 40],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
    # Checkpoints
    parser.add_argument('--save', '-s', type=str, default=None, help='Folder to save checkpoints.')
    parser.add_argument('--save_steps', '-ss', type=int, default=200, help='steps to save checkpoints.')
    parser.add_argument('--load', '-l', type=str, help='Checkpoint path to resume / test.')
    parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
    # Architecture
    parser.add_argument('--depth', type=int, default=29, help='Model depth.')
    parser.add_argument('--cardinality', type=int, default=16, help='Model cardinality (group).')
    parser.add_argument('--base_width', type=int, default=64, help='Number of channels in each group.')
    parser.add_argument('--widen_factor', type=int, default=4, help='Widen factor. 4 -> 64, 8 -> 128, ...')
    # Acceleration
    parser.add_argument('--gpu', type=int, default=1, help='0 = CPU.')
    parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default=None, help='Log folder.')

    parser.add_argument('--crop_size', type=int, default=320, help='The size of image.')
    parser.add_argument('--aug', type=str, default='crop', help='The size of image.')

    parser.add_argument('--display', type=int, default=0, help='display or not')
    parser.add_argument('--tensorboard_log', type=int, default=5, help='display or not')

    #####
    # pix2pixHD config
    parser.add_argument('--norm', type=str, default='instance',
                        help='instance normalization or batch normalization')
    parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
    parser.add_argument('--data_type', default=32, type=int, choices=[8, 16, 32],
                        help="Supported data type i.e. 8, 16, 32 bit")
    parser.add_argument('--verbose', action='store_true', default=False, help='toggles verbose')
    parser.add_argument('--fp16', action='store_true', default=False, help='train with AMP')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

    # input/output sizes
    parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
    parser.add_argument('--loadSize', type=int, default=1024, help='scale images to this size')
    parser.add_argument('--fineSize', type=int, default=512, help='then crop to this size')
    parser.add_argument('--label_nc', type=int, default=3, help='# of input label channels')
    parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
    parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')

    # for setting inputs
    parser.add_argument('--dataroot', type=str, default='./datasets/cityscapes/')
    parser.add_argument('--resize_or_crop', type=str, default='scale_width',
                        help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
    parser.add_argument('--serial_batches', action='store_true',
                        help='if true, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
    parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                        help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

    # for generator
    parser.add_argument('--netG', type=str, default='global', choices=['global', 'local'], help='selects model to use for netG')
    parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
    parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
    parser.add_argument('--n_blocks_global', type=int, default=9,
                        help='number of residual blocks in the global generator network')
    parser.add_argument('--n_blocks_local', type=int, default=3,
                        help='number of residual blocks in the local enhancer network')
    parser.add_argument('--n_local_enhancers', type=int, default=1, help='number of local enhancers to use')

    parser.add_argument('--niter_fix_global', type=int, default=0,
                        help='number of epochs that we only train the outmost local enhancer')

    # for instance-wise features
    parser.add_argument('--use_instance', default=1, type=int,
                        help='if true, do add instance map as input')
    parser.add_argument('--instance_feat', action='store_true',
                        help='if specified, add encoded instance features as input')
    parser.add_argument('--label_feat', action='store_true',
                        help='if specified, add encoded label features as input')
    parser.add_argument('--feat_num', type=int, default=3, help='vector length for encoded features')
    parser.add_argument('--load_features', action='store_true', help='if specified, load precomputed feature maps')
    parser.add_argument('--n_downsample_E', type=int, default=4, help='# of downsampling layers in encoder')
    parser.add_argument('--nef', type=int, default=16, help='# of encoder filters in the first conv layer')
    parser.add_argument('--n_clusters', type=int, default=10, help='number of clusters for features')

    parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')

    # for discriminators
    parser.add_argument('--num_D', type=int, default=3, help='number of discriminators to use')
    parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
    parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
    parser.add_argument('--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')

    parser.add_argument('--use_ganFeat_loss', default=1, type=int,
                        help='if true, use discriminator feature matching loss')

    parser.add_argument('--use_vgg_loss', default=1, type=int,
                        help='if true, use VGG feature matching loss')
    parser.add_argument('--vgg_type', default='vgg16', choices=['vgg16', 'vgg19'],
                        help='if none, do not use VGG feature matching loss')

    parser.add_argument('--use_lsgan', default=1, type=int,
                        help='if true, use least square GAN, if false, use vanilla GAN')

    parser.add_argument('--use_low_level_loss', default=0, type=int, choices=[0, 1],
                        help='use low level loss or not, used in img2map')

    parser.add_argument('--low_level_loss', default='L1', type=str, choices=['L1', 'L2', 'smoothL1'],
                        help='low level loss, used in img2map')

    args = parser.parse_args()
    for arg in vars(args):
        print(arg, getattr(args, arg))
    if args.save is None:
        args.save = f'../../temp_run'
    if args.log is None:
        args.log = args.save

    args.tensorboard_path = osp.join(args.save, 'tensorboard')
    create_dir(args.tensorboard_path)
    args.scheduler = f'{args.optimizer}_{args.scheduler}'
    return args

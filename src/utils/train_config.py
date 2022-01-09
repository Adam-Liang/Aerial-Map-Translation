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
import torch
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
import time
from tqdm import tqdm
from torch import optim
from torch.utils.data import DataLoader


def config():
    parser = argparse.ArgumentParser(description='Trains GAN on CIFAR',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # Positional arguments
    parser.add_argument('--cifar_10_data_path', type=str, default='/Users/chenlinwei/dataset/cifar-10-batches-py',
                        help='Root for the Cifar dataset.')
    parser.add_argument('--mnist_data_path', type=str, default='/Users/chenlinwei/dataset',
                        help='Root for the MNIST dataset.')
    parser.add_argument('--voc2012_data_path', type=str, default='/Users/chenlinwei/dataset/VOCdevkit/VOC2012',
                        help='Root for the voc dataset.')
    parser.add_argument('--voc_repeat', type=int, default=10, help='Repeat for the voc dataset in train_voc_weakly.py')
    parser.add_argument('--sbd_repeat', type=int, default=1, help='Repeat for the voc dataset in train_voc_weakly.py')
    parser.add_argument('--sbd_data_path', type=str, default='/Users/chenlinwei/dataset/SBD_FULL11355/dataset',
                        help='Root for the voc dataset.')
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
    parser.add_argument('--epochs', '-e', type=int, default=50, help='Number of epochs to train.')
    parser.add_argument('--batch_size', '-b', type=int, default=2, help='Batch size.')
    parser.add_argument('--lr', type=float, default=0.0001, help='The Learning Rate.')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='The Learning Rate if generator.')
    parser.add_argument('--d_lr', type=float, default=0.0004, help='The Learning Rate of discriminator.')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--test_batch_size', type=int, default=10)
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
    parser.add_argument('--prefetch', type=int, default=12, help='Pre-fetching threads.')
    # i/o
    parser.add_argument('--log', type=str, default=None, help='Log folder.')

    parser.add_argument('--crop_size', type=int, default=320, help='The size of image.')
    parser.add_argument('--aug', type=str, default='resize', help='The size of image.')

    parser.add_argument('--display', type=int, default=0, help='display or not')
    parser.add_argument('--display_iter', type=int, default=3, help='display interval')
    parser.add_argument('--img_channel', type=int, default=3, help='channels of training image')
    parser.add_argument('--dcgan', type=str, default='mnist', help='channels of training image')
    args = parser.parse_args()
    if args.save is None:
        args.save = f'../../temp_run'
    if args.log is None:
        args.log = args.save
    args.scheduler = f'{args.optimizer}_{args.scheduler}'
    return args

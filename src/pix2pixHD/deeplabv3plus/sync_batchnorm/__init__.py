# -*- coding: utf-8 -*-
# File   : __init__.py
# Author : Jiayuan Mao
# Email  : maojiayuan@gmail.com
# Date   : 27/01/2018
#
# This file is part of Synchronized-BatchNorm-PyTorch.
# https://github.com/vacancy/Synchronized-BatchNorm-PyTorch
# Distributed under MIT License.

from src.pix2pixHD.deeplabv3plus.sync_batchnorm.batchnorm import SynchronizedBatchNorm1d, SynchronizedBatchNorm2d, SynchronizedBatchNorm3d
from src.pix2pixHD.deeplabv3plus.sync_batchnorm.batchnorm import patch_sync_batchnorm, convert_model
from src.pix2pixHD.deeplabv3plus.sync_batchnorm.replicate import DataParallelWithCallback, patch_replication_callback

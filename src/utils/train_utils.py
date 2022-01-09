# -*- coding: utf-8 -*-

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim
import os
from os import path as osp
import shutil

# from logger import Logger, ModelSaver
# from model import get_model
from torchvision import transforms


def mean(l):
    return sum(l) / float(len(l))


def save_safely(file, dir_path, file_name):
    """
    save the file safely, if detect the file name conflict,
    save the new file first and remove the old file
    """
    if not osp.exists(dir_path):
        os.mkdir(dir_path)
        print('*** dir not exist, created one')
    save_path = osp.join(dir_path, file_name)
    if osp.exists(save_path):
        temp_name = save_path + '.temp'
        torch.save(file, temp_name)
        os.remove(save_path)
        os.rename(temp_name, save_path)
        print('*** find the file conflict while saving, saved safely')
    else:
        torch.save(file, save_path)


def save_checkpoint(args, state, is_best, iou, name='checkpoint.pth'):
    print('===> saving checkpoint')
    save_safely(file=state, dir_path=args.save, file_name=name)
    # torch.save(state, file_name)
    if is_best:
        file_name = os.path.join(args.save, name)
        best_path = os.path.join(args.save, f'best_iou_model_{iou:.4f}.pth')
        shutil.copyfile(file_name, best_path)
        print(f'===> Saving the best model:{iou:.4f} as {best_path}')


def get_device(args):
    return torch.device('cuda' if args.gpu >= 1 and torch.cuda.is_available() else 'cpu')


def load(args, model, optimizer=None, logger=None):
    # resume
    start_epoch = 0
    best_iou = 0
    print('Resuming model from {}'.format(args.resume_path))
    try:
        state = torch.load(args.resume_path, map_location='cpu')
        start_epoch = state['epoch'] + 1
        best_iou = state['best_iou']
        model.load_state_dict(state['model_state_dict'], strict=True)
        if optimizer is not None:
            optimizer.load_state_dict(state['optim_state_dict'])
        if logger is not None:
            logger.data = state['logger_data']
    except Exception:
        print('load {} fail!'.format(args.resume_path))

    if args.epoch_update is not None:
        start_epoch = args.epoch_update
        print(f'===> Epochs update to {start_epoch}')

    return start_epoch, best_iou, model, optimizer, logger


def model_accelerate(args, model):
    r"""
    Use it with a provided, customized data parallel wrapper:

    from sync_batchnorm import SynchronizedBatchNorm1d, DataParallelWithCallback

    sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
    sync_bn = DataParallelWithCallback(sync_bn, device_ids=[0, 1])
    Or, if you are using a customized data parallel module, you can use this library as a monkey patching.

    from torch.nn import DataParallel  # or your customized DataParallel module
    from sync_batchnorm import SynchronizedBatchNorm1d, patch_replication_callback

    sync_bn = SynchronizedBatchNorm1d(10, eps=1e-5, affine=False)
    sync_bn = DataParallel(sync_bn, device_ids=[0, 1])
    patch_replication_callback(sync_bn)  # monkey-patching
    You can use convert_model to convert your model to use Synchronized BatchNorm easily.

    from torchvision import models
    from sync_batchnorm import convert_model

    m = models.resnet18(pretrained=True)
    m = convert_model(m)
    :param args:
    :param model:
    :return:
    """
    from model.sync_batchnorm.replicate import patch_replication_callback, DataParallelWithCallback
    from model.sync_batchnorm import convert_model
    if torch.cuda.device_count() > 0 and args.gpu > 0:
        model = convert_model(model)
        model = torch.nn.DataParallel(model)
        patch_replication_callback(model)
        device = get_device(args)
        model = model.to(device)
        print(f'*** {model.__class__.__name__} to GPUs, syncbatch OK.')
    else:
        model = nn.DataParallel(model)
    return model


def get_params(model, key):
    for m in model.named_modules():
        # print(m)
        if key == '1x':
            if (any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1], nn.Conv2d):
                for p in m[1].parameters():
                    yield p
        elif key == '10x':
            if (not any([(i in m[0]) for i in ('pretrained_net', 'encoder', 'backbone')])) and isinstance(m[1],
                                                                                                          nn.Conv2d):
                for p in m[1].parameters():
                    yield p


def get_optimizer(args, model):
    name = args.optimizer
    assert isinstance(name, str)
    if name.lower() == 'adam':
        return optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    elif name.lower() == 'sgd':
        return optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)
    elif name.lower() == 'adam_finetune':
        try:
            optimizer = optim.Adam(params=[
                {'params': get_params(model.module, key='1x'), 'lr': 0.1 * args.lr},
                {'params': get_params(model.module, key='10x'), 'lr': args.lr}]
            )
        except:
            optimizer = optim.Adam(params=[
                {'params': get_params(model, key='1x'), 'lr': 0.1 * args.lr},
                {'params': get_params(model, key='10x'), 'lr': args.lr}]
            )
        return optimizer
    elif name.lower() == 'sgd_finetune':
        try:
            optimizer = optim.SGD(
                params=[
                    {'params': get_params(model.module, key='1x'), 'lr': 0.1 * args.lr},
                    {'params': get_params(model.module, key='10x'), 'lr': args.lr}
                ],
                momentum=args.momentum,
                weight_decay=4e-5
            )
        except:
            optimizer = optim.SGD(
                params=[
                    {'params': get_params(model, key='1x'), 'lr': 0.1 * args.lr},
                    {'params': get_params(model, key='10x'), 'lr': args.lr}
                ],
                momentum=args.momentum,
                weight_decay=4e-5
            )
        return optimizer
    else:
        raise Exception(f'*** optimizer:{name} not implemented!')
    # assert name.lower() in [op.lower() for op in op_dict.keys()]
    # return op_dict.get(name)


def print_lr(optimizer):
    if isinstance(optimizer, nn.DataParallel):
        optimizer = optimizer.module
    for param_group in optimizer.param_groups:
        print('===> learning rate: {}'.format(param_group['lr']))


def get_lr(optimizer):
    if isinstance(optimizer, nn.DataParallel):
        optimizer = optimizer.module
    for param_group in optimizer.param_groups:
        return param_group['lr']


def update_lr(module, lr: float, mute=False):
    if isinstance(module, nn.DataParallel):
        module = module.module
    try:
        module.param_groups[0]['lr'] = 0.1 * lr
        module.param_groups[1]['lr'] = lr
        info = '====> update lr-0:{:.12f} | lr-1:{:.12f} complete.'.format(module.param_groups[0]['lr'],
                                                                           module.param_groups[1]['lr'])
        if not mute:
            print(info)
        return info
    except:
        for param_group in module.param_groups:
            # print(param_group)
            param_group['lr'] = lr
        info = f'====> update lr:{lr:.12f} complete.'
        if not mute:
            print(info)
        return info


def get_sheduler(args, optimizer):
    def lambda_(epoch):
        return pow((1 - ((epoch) / args.epochs)), 0.9)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda_, )
    return scheduler


def poly_adjust_lr(optimizer, lr, itr, max_itr, power=0.9, mute=False):
    now_lr = lr * (1 - itr / (max_itr + 1)) ** power
    info = update_lr(optimizer, now_lr, mute=mute)
    return info


def freeze_batch_norm(model):
    def freeze_one(m):
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            m.eval()
            # print(m)

    # if isinstance(model, nn.DataParallel):
    #     model = model.module
    model.apply(freeze_one)


def check_batch_norm_freeze(model):
    if isinstance(model, nn.DataParallel):
        model = model.module

    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, SynchronizedBatchNorm2d):
            if m.training is not False:
                print('*** BN of model do not freeze!')
                return
            # for parameter in m.parameters():
            #     if parameter.requires_grad != False:
            #         print('*** BN of model do not freeze!')
            #         return
    print('*** BN of model is freezed')


def _test_freeze_batch_norm():
    from train_config import get_config
    from models import get_model
    from models.sync_batchnorm import convert_model
    args = get_config()
    model = get_model(args)
    model = convert_model(model)
    for m in model._modules:
        print(m)
    model = nn.DataParallel(model)
    freeze_batch_norm(model)
    check_batch_norm_freeze(model)


def copy_weakly_model(args, model):
    from models import get_weakly_model
    _model = get_weakly_model(args).to(get_device(args))
    _model = nn.DataParallel(_model)
    _model.load_state_dict(model.state_dict())
    return _model


if __name__ == '__main__':
    _test_freeze_batch_norm()

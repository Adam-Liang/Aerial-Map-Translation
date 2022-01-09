import torch
from torch import optim


def get_lr_lambda(total_epochs=200):
    half_epochs = total_epochs // 2
    # plus one to avoid zero lr
    lr_lambda = lambda epoch: 1.0 if epoch < half_epochs \
        else float(total_epochs - epoch + 1) / float(total_epochs - half_epochs + 1)
    return lr_lambda


def get_lr(optimizer):
    if isinstance(optimizer, torch.nn.DataParallel):
        optimizer = optimizer.module
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_hinge_scheduler(args, optimizer):
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_lambda(args.epochs))
    return scheduler


if __name__ == '__main__':
    model = torch.nn.Linear(2, 2)
    adam = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = optim.lr_scheduler.LambdaLR(adam, lr_lambda=get_lr_lambda())

    for i in range(200):
        adam.step()
        scheduler.step()
        print(get_lr(adam))

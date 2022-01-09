import os
import torch

def check_program(floder): #todo：此处应设为与模型有关
    files = ["opt.json", "log.json", "state.json"]
             # "modeldata/latest_netD.pth", "modeldata/latest_netG.pth",
             # "modeldata/latest_optimizerD.pth", "modeldata/latest_optimizerG.pth"]
    for filename in files:
        assert os.path.isfile(os.path.join(floder,filename)),"文件夹格式有误！未找到[%s]"%os.path.join(floder,filename)

def set_gpuids(opt):
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)

    # set gpu ids
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

def get_inner_path(file_path,floder_path):
    assert file_path[:len(floder_path)]==floder_path,"传入的文件不在文件夹中！[%s][%s]"%(file_path,floder_path)
    file_path=file_path[len(floder_path)+1:]
    return file_path


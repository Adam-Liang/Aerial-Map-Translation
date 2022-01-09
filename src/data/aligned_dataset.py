import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
from util.my_util import get_inner_path
import cv2
import numpy as np
from tqdm import tqdm


def make_align_img(dir_A,dir_B,dir_AB):
    print("Align data floder creating!")
    num=0
    imgs_A=sorted(make_dataset(dir_A))
    imgs_B=sorted(make_dataset(dir_B))
    imgs_A_=[]
    imgs_B_=[]
    for img_A in imgs_A:
        imgs_A_.append(os.path.splitext(img_A)[0])
    for img_B in imgs_B:
        imgs_B_.append(os.path.splitext(img_B)[0])
    for i in range(len(imgs_A)):
        img_A=imgs_A[i]
        img_inner = get_inner_path(img_A, dir_A)
        if get_inner_path(imgs_A_[i], dir_A) == get_inner_path(imgs_B_[i],dir_B):
            photo_A = cv2.imread(img_A)
            photo_B = cv2.imread(imgs_B[i])
            if photo_A.shape == photo_B.shape:
                photo_AB = np.concatenate([photo_A, photo_B], 1)
                img_AB = os.path.join(dir_AB, os.path.splitext(img_inner)[0]+'.png')
                if not os.path.isdir(os.path.split(img_AB)[0]):
                    os.makedirs(os.path.split(img_AB)[0])
                cv2.imwrite(img_AB, photo_AB)
                num += 1
    # for img_A in tqdm(imgs_A):
    #     img_inner=get_inner_path(img_A,dir_A)
    #     if os.path.join(dir_B,img_inner) in imgs_B:
    #         photo_A=cv2.imread(img_A)
    #         photo_B=cv2.imread(os.path.join(dir_B,img_inner))
    #         if photo_A.shape==photo_B.shape:
    #             photo_AB=np.concatenate([photo_A, photo_B], 1)
    #             img_AB=os.path.join(dir_AB,img_inner)
    #             if not os.path.isdir(os.path.split(img_AB)[0]):
    #                 os.makedirs(os.path.split(img_AB)[0])
    #             cv2.imwrite(img_AB, photo_AB)
    #             num+=1
    print("Align data floder created! %d img was processed"%num)


class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        assert (os.path.isdir(self.dir_AB) or (os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'))),"dataset path not exsit! [%s]"%self.dir_AB
        if (not os.path.isdir(self.dir_AB)) and os.path.isdir(self.dir_AB+'A') and os.path.isdir(self.dir_AB+'B'):#to do:数据集AB自动合并功能尚不支持英文路径
            os.makedirs(self.dir_AB)
            dir_A=self.dir_AB+'A'
            dir_B=self.dir_AB+'B'
            make_align_img(dir_A,dir_B,self.dir_AB)

        self.AB_paths = sorted(make_dataset(self.dir_AB))  #make_dataset:返回一个所有图像格式文件（含路径）的list
        assert (len(self.AB_paths)!=0),"dataset floder is empty! [%s]"%self.dir_AB

        assert(opt.resize_or_crop == 'resize_and_crop')

        transform_list = [transforms.ToTensor(),  # Totensor操作会将img/ndarray转化至0~1的FloatTensor
                          transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5))]  # mean ,std;  result=(x-mean)/std

        self.transform = transforms.Compose(transform_list)  # 串联组合

        #deeplabv3+相关
        self.dir_segs=os.path.join(opt.dataroot, opt.phase+"_seg")
        from src.util.make_seg_img import labelpixels
        flag=0
        if os.path.isdir(self.dir_segs):
            segs = make_dataset(self.dir_segs)
            if len(self.AB_paths) == len(segs):
                flag=1
            else:
                import shutil
                shutil.rmtree(self.dir_segs)
        if flag==0:
            os.makedirs(self.dir_segs)
            for map in self.AB_paths:
                map_np_AB = np.array(Image.open(map).convert("RGB"))
                map_np=map_np_AB[:,map_np_AB.shape[1]//2:,:]
                seg_np = labelpixels(map_np).astype(np.uint8)
                seg_pil = Image.fromarray(seg_np)
                seg_path = os.path.join(self.dir_segs, get_inner_path(map, self.dir_AB))
                seg_pil.save(seg_path)
        self.seg_paths=sorted(make_dataset(self.dir_segs))

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.opt.loadSize * 2, self.opt.loadSize), Image.BICUBIC)
        AB = self.transform(AB)

        w_total = AB.size(2)
        w = int(w_total / 2)
        h = AB.size(1)
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = AB[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        B = AB[:, h_offset:h_offset + self.opt.fineSize,
               w + w_offset:w + w_offset + self.opt.fineSize]

        # deeplabv3相关，取label图，之后需将label图与A，B同步处理
        seg_path = self.seg_paths[index]
        seg = Image.open(seg_path)
        seg = np.asarray(seg)
        seg = torch.from_numpy(seg)  # H W
        seg = seg[h_offset:h_offset + self.opt.fineSize,
              w_offset:w_offset + self.opt.fineSize]

        if False: #self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        # self.opt.no_flip=False
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)  # 64bit带符号整型，why？因为index_select要求下标为longtensor类型
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            seg=seg.index_select(1,idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)



        # deeplabv3相关-为迎合使用的pretrain模型，需将标准化参数改为ImageNet版本
        toseg_transform_list = [transforms.Normalize((0, 0, 0),(2, 2, 2)),  # mean ,std;  result=(x-mean)/std
                                transforms.Normalize((-0.5,-0.5,-0.5),(1,1,1)),
                                transforms.Normalize((0.485,0.456,0.406),((0.229,0.224,0.225)))] # 恢复标准化前的数值，并换一组数据标准化
        toseg_transform = transforms.Compose(toseg_transform_list)
        A_seg=toseg_transform(A)
        B_seg=toseg_transform(B)

        return {'A': A, 'B': B, 'seg':seg,'A_seg':A_seg,'B_seg':B_seg,
                'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'AlignedDataset'

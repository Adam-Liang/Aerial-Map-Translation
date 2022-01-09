import torch
import numpy as np
from PIL import Image
import os
import os.path as osp
import cv2
import math
import random

def hsv2rgb(h,s=1.0,v=1.0): #R, G, B是 [0, 255]. H 是[0, 360]. S, V 是 [0, 1].
    h = float(h)
    s = float(s)
    v = float(v)
    h60 = h / 60.0
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return r, g, b

def visual_theta_rho(seg_vis, road_mask, seg_theta, seg_rho, seg_roadwid):
    if len(seg_vis.shape)==2:
        seg_vis=gray2rgb(seg_vis)
    # 做一个可视化，带图例的图，以256*256的图为例，rho取值范围为362*2，theta取值范围为-90~90度
    color_card_len=200
    thetas = np.deg2rad(np.arange(-90.0, 90.0, 1))
    thetas_dict=dict(zip(thetas,range(len(thetas))))
    thetas_card=[]
    rhos_card=[]
    roadwid_card = []
    if isinstance(seg_theta,np.ndarray):
        for i in range(len(thetas)):
            h=int(360*i/len(thetas))
            r,g,b=hsv2rgb(h,1.0,1.0)
            thetas_card.append([r,g,b])
    if isinstance(seg_rho, np.ndarray):
        for i in range(362*2):
            h = int(360 * i / (362*2))
            r, g, b = hsv2rgb(h, 1.0, 1.0)
            rhos_card.append([r,g,b])
    if isinstance(seg_roadwid, np.ndarray):
        for i in range(seg_roadwid.max()+1):
            h = int(240 * i / (seg_roadwid.max()+1))
            r, g, b = hsv2rgb(h, 1.0, 1.0)
            roadwid_card.append([r,g,b])
    y_idxs, x_idxs = np.nonzero(road_mask)
    thetas_vis=np.full(seg_vis.shape,255,np.uint8)  # 256*256*3
    rhos_vis=np.full(seg_vis.shape,255,np.uint8)
    roadwid_vis = np.full(seg_vis.shape, 255, np.uint8)
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        if isinstance(seg_theta, np.ndarray):
            if seg_theta[y,x]==-2.0:
                thetas_vis[y, x] = [0,0,0]
            else:
                thetas_vis[y,x]=thetas_card[thetas_dict[seg_theta[y,x]]]
        if isinstance(seg_rho, np.ndarray):
            rhos_vis[y,x]=rhos_card[seg_rho[y,x]]
        if isinstance(seg_roadwid, np.ndarray):
            roadwid_vis[y,x]=roadwid_card[seg_roadwid[y,x]]
    # 做两个色卡条子
    thetas_card_vis=np.zeros((50,200,3),dtype=np.uint8)
    rhos_card_vis=np.zeros((50,200,3),dtype=np.uint8)
    roadwid_card_vis = np.zeros((50, 200, 3), dtype=np.uint8)
    for i in range(200):
        for j in range(50):
            if isinstance(seg_theta, np.ndarray):
                thetas_card_vis[j][i]=thetas_card[int(len(thetas_card)*i/200)]
            if isinstance(seg_rho, np.ndarray):
                rhos_card_vis[j][i]=rhos_card[int(len(rhos_card)*i/200)]
            if isinstance(seg_roadwid, np.ndarray):
                roadwid_card_vis[j][i]=roadwid_card[int(len(roadwid_card)*i/200)]
    # ret=np.full((356,256*4,3),255,dtype=np.uint8)
    # ret[24:74,256:456,:]=thetas_card_vis
    # ret[24:74, 512:712, :] = rhos_card_vis
    # ret[24:74, 768:968, :] = roadwid_card_vis
    # ret[100:356,0:256,:]=seg_vis
    # ret[100:356, 256:512, :] = thetas_vis
    # ret[100:356, 512:768, :] = rhos_vis
    # ret[100:356, 768:1024, :] = roadwid_vis
    ret = np.full((356, 256 * 3, 3), 255, dtype=np.uint8)
    ret[24:74, 256:456, :] = thetas_card_vis
    ret[24:74, 512:712, :] = roadwid_card_vis
    ret[100:356, 0:256, :] = seg_vis
    ret[100:356, 256:512, :] = thetas_vis
    ret[100:356, 512:768, :] = roadwid_vis
    return ret

def pred2gray(pred): # pred: tensor:bs*n_class*h*w
    bs,n_class,h,w=pred.size()
    pred=pred.data.cpu().numpy()
    gray = pred.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
    gray=torch.from_numpy(gray)
    return gray

def pred2gray_bytensor(pred): # pred: tensor:bs*n_class*h*w
    # 为使梯度能够反向传播，不经numpy中转。
    bs,n_class,h,w=pred.size()
    gray=pred.argmax(axis=1)
    # pred=pred.data.cpu().numpy()
    # gray = pred.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(bs, h, w)
    # gray=torch.from_numpy(gray)
    return gray

def gray2rgb(gray,n_class=5,label_list=[[248,249,250], [253,226,147], [156,192,249],[206,234,214],[214,217,220]]): # gray: np:h*w
# def gray2rgb(gray,n_class=5,label_list=[[239,238,236],[255,255,255],[170,218,255],[208,236,208],[255,255,255]]): # gray: np:h*w
    h,w=gray.shape
    mask=[]
    rgb=np.zeros((h,w,3))
    for i in range(n_class):
        tmp=(gray==i)
        mask.append(tmp)
        rgb+=np.expand_dims(tmp,2).repeat(3,axis=2)*label_list[i]
    rgb=rgb.astype(np.uint8)
    return rgb

def seed_torch(seed):
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.enabled = False


if __name__=='__main__':
    path="D:\\map_translate\\数据集\\20191117第三批数据\\sample_20191115\\taiwan_10_18_20191115_copy\\15_tiny\\test_seg\\15-7022-13708.png"
    gray=Image.open(path)
    gray=np.array(gray)
    rgb=gray2rgb(gray)
    rgb=Image.fromarray(rgb)
    rgb.save("D:\\map_translate\\数据集\\20191117第三批数据\\sample_20191115\\taiwan_10_18_20191115_copy\\15_tiny\\test_seg\\15-7022-13708___.png")
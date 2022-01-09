import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target): #input: bs,c,h,w    target: bs h w
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C  # contiguous():类似深拷贝，开辟新的内存空间并调整实际存储形式
        target = target.view(-1,1) # N*H*W,1

        logpt = F.log_softmax(input,dim=1)
        logpt = logpt.gather(1,target) #torch.gather(input, dim, index, out=None) 效果待确认，不太好理解 # 目标为取出正确类别对应的logpt
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            # if self.alpha.type()!=input.data.type(): # 因为cuda导致bug，忽略该步
            #     self.alpha = self.alpha.type_as(input.data)
            # print(target.max().cpu())
            assert (target<5).all()
            assert (target > -1).all()
            # print(target.data.cpu().max())
            # print(target.data.cpu().min())
            at = self.alpha.gather(0,target.data.cpu().view(-1)) # 将所有pix的权重列出（根据其真实label）
            logpt = logpt * Variable(at.type_as(logpt))

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()
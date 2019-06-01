import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import log_sum_exp, focal_sum_exp, decode

class SmoothL1_Mixup_Balance_loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=1.5, balance = False, mixup = False, size_average=False):
        super(SmoothL1_Mixup_Balance_loss,self).__init__()
        self.balance = balance
        self.mixup = mixup
        self.size_average = size_average
        if self.balance:
            self.a = alpha
            self.r = gamma
            self.b = math.exp(gamma / alpha) - 1
            self.c = gamma / self.b - alpha

    def forward(self, predict, truth, weight=None):
        if self.mixup:
            assert predict.shape[0]== truth.shape[0]== weight.shape[0]
        else:
            assert predict.shape[0] == truth.shape[0]
        t = torch.abs(truth-predict)
        if self.balance:
            smbloss = torch.where(t < 1, self.a * (self.b * t + 1) * torch.log(self.b * t + 1) / self.b - self.a * t, self.r * t + self.c)
        else:
            smbloss = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        if self.mixup:
            smbloss = smbloss.sum(1, keepdim=True) * weight
        else:
            smbloss = smbloss.sum(1)
        if self.size_average:
            return torch.mean(smbloss)
        else:
            return smbloss.sum()

class Crossentropy_Mixup_SoftmaxFocal_LableSmooth_loss(nn.Module):
    def __init__(self, mixup=False, focal_loss=False, gamma=2, alpha=1, label_smooth=False,size_average=False):
        super(Crossentropy_Mixup_SoftmaxFocal_LableSmooth_loss,self).__init__()
        self.mixup = mixup
        self.softmax_focal = focal_loss
        if self.softmax_focal:
            self.gamma = gamma
            self.alpha = alpha
        self.label_smooth = label_smooth
        self.size_average = size_average

    def forward(self, predict, truth, weight=None):
        if self.mixup:
            assert predict.shape[0] == truth.shape[0] == weight.shape[0]
        else:
            assert predict.shape[0] == truth.shape[0]
        if self.softmax_focal:
            # using OHEM and focal loss with CE
            soft_score = focal_sum_exp(predict)
            pro = self.alpha * (1 - soft_score) ** self.gamma
            cmsloss = (log_sum_exp(predict) - predict.gather(1, truth.view(-1, 1))) * pro.gather(1, truth.view(-1,1))
        elif self.label_smooth:
            cmsloss = (log_sum_exp(predict, label_smooth=True) * truth).sum(1, keepdim=True)
        else:
            cmsloss = log_sum_exp(predict) - predict.gather(1, truth.view(-1, 1))
        if self.mixup:
            cmsloss = cmsloss * weight
        if self.size_average:
            return cmsloss.mean()
        else:
            return cmsloss.sum()

class GIoUloss(nn.Module):
    def __init__(self,size_average=False):
        super(GIoUloss,self).__init__()
        self.size_average = size_average

    def _GIoU(self, p, g):
        areas_p = (p[:, 2] - p[:, 0]) * (p[:, 3] - p[:, 1])
        areas_g = (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1])
        x1y1 = torch.max(p[:, :2], g[:, :2])
        x2y2 = torch.min(p[:, 2:], g[:, 2:])
        inter = torch.clamp((x2y2 - x1y1), min=0)
        area_inter = inter[:, 0] * inter[:, 1]
        x1y1 = torch.min(p[:, :2], g[:, :2])
        x2y2 = torch.max(p[:, 2:], g[:, 2:])
        total = x2y2 - x1y1
        area_total = total[:, 0] * total[:, 1]  # 闭包区域面积
        uni = areas_g + areas_p - area_inter
        iou = area_inter / uni
        Giou = iou - (area_total - uni) / area_total
        return Giou

    def forward(self, predict, priors, target, variance=[0.1,0.2]):
        assert priors.shape == predict.shape == target.shape, "GIoU loss ERROR!"

        p = decode(predict, priors, variance)
        loss = 1 - self._GIoU(p, target)
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


if __name__ == "__main__":
    print("This is a loss function implementation file.")
    pass

import numpy as np
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, match_mixup,point_form
GPU = False
if torch.cuda.is_available():
    GPU = True
from .loss import *

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """


    def __init__(self, num_classes,overlap_thresh,prior_for_matching,bkg_label,neg_mining,neg_pos,neg_overlap,encode_target,
                 label_smmooth=False, balance_l1=False, focal_loss=False, giou=False):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching  = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1,0.2]
        self.label_smooth = label_smmooth
        if self.label_smooth:
            self.label_pos = 0.9
            self.label_neg = (1.0 - self.label_pos) / (self.num_classes - 1)
        self.balance_l1 = balance_l1
        self.focal_loss = focal_loss
        self.softmax_focal = False # using OHEM, CEWithsoftmax and Focal loss
        self.sigmoid_focal = False # Original Focal loss(Using sigmoid with CE)
        if self.focal_loss:
            self.softmax_focal = True
        if self.sigmoid_focal:
            self.alpha = 0.25
            self.gamma = 2.0
        self.giou = giou

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))
        num_classes = self.num_classes

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        if targets[0].shape[1] == 6:# mixup
            weight_t = torch.Tensor(num, num_priors)
        for idx in range(num):
            defaults = priors.data
            if targets[idx].shape[1] == 6:  # mixup
                truths = targets[idx][:, :-2].data
                labels = targets[idx][:, -2].data
                weight_loss = targets[idx][:, -1].data
                match_mixup(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx, weight_t, weight_loss, self.giou)
            elif targets[idx].shape[1] == 5:  # no moxiup
                truths = targets[idx][:, :-1].data
                labels = targets[idx][:, -1].data
                match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx, self.giou)
            else:
                print('The shape of targets is error')

        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,requires_grad=False)

        pos = conf_t > 0

        mix_up = (False, True)[targets[0].shape[1] == 6]
        pos_weight = None
        weights_conf = None

        # Localization Loss (Smooth L1)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)

        if self.giou:
            # prior_giou = point_form(priors)  # [x,y,h,w]->[x0,y0,x1,y1]
            prior_giou = priors.unsqueeze(0).expand(num, num_priors, 4)
            prior_giou = prior_giou[pos_idx].view(-1, 4)
            reg_loss = GIoUloss()
            loss_l = reg_loss(loc_p, prior_giou, loc_t)
        else:
            if mix_up:
                weight_t = weight_t.cuda()
                weight_t = Variable(weight_t, requires_grad=False)
                pos_weight = weight_t[pos].view(-1, 1)

            reg_loss = SmoothL1_Mixup_Balance_loss(mixup=mix_up, balance=self.balance_l1, size_average=False)
            loss_l = reg_loss(loc_p, loc_t, pos_weight)

        # Confidence Loss
        if self.sigmoid_focal:
            # if use original focal loss, please modify the output of the test in models/SSD.py to the sigmoid
            batch_conf = conf_data.view(-1, self.num_classes)
            label_onehot = batch_conf.clone().zero_().scatter(1, conf_t.view(-1,1), 1)
            alpha = self.alpha * label_onehot + (1 - self.alpha) * (1 - label_onehot)
            p = torch.sigmoid(batch_conf)
            pt = torch.where(label_onehot==1, p, 1-p)
            loss_c = - alpha * ((1 - pt) ** self.gamma) * torch.log(pt)
            loss_c = loss_c.sum()
            num_pos = pos.long().sum(1, keepdim=True)
        else:
            batch_conf = conf_data.view(-1, self.num_classes)
            loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

            # Hard Negative Mining
            loss_c[pos.view(-1, 1)] = 0  # filter out pos boxes for now
            loss_c = loss_c.view(num, -1)
            _, loss_idx = loss_c.sort(1, descending=True)
            _, idx_rank = loss_idx.sort(1)
            num_pos = pos.long().sum(1, keepdim=True)
            num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
            neg = idx_rank < num_neg.expand_as(idx_rank)

            # Confidence Loss Including Positive and Negative Examples
            pos_idx = pos.unsqueeze(2).expand_as(conf_data)
            neg_idx = neg.unsqueeze(2).expand_as(conf_data)
            conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
            if self.label_smooth:
                p = conf_t.clone().view(-1, 1).float()
                lp = torch.where(p < 1, p + 1, torch.tensor(self.label_pos).cuda())
                label = batch_conf.clone().zero_().scatter_(1, conf_t.view(-1, 1), lp)
                label[:, 1:][pos.clone().view(-1, 1).flatten()] += self.label_neg
                label_ohem = (pos + neg).view(-1, 1).expand_as(batch_conf)
                targets_weighted = label[label_ohem.gt(0)].view(-1, self.num_classes)
            else:
                targets_weighted = conf_t[(pos + neg).gt(0)]
            if mix_up:
                weights_conf = weight_t[(pos + neg).gt(0)]
                weights_conf = torch.where(weights_conf > 0, weights_conf, weights_conf + 1.0).view(-1, 1)

            conf_loss = Crossentropy_Mixup_SoftmaxFocal_LableSmooth_loss(mixup=mix_up,focal_loss=self.softmax_focal,gamma=2.0,alpha=1.0,
                                                                                 label_smooth=self.label_smooth,size_average=False)
            loss_c = conf_loss(conf_p, targets_weighted, weights_conf)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = max(num_pos.data.sum().float(), 1)
        loss_l/=N
        loss_c/=N
        return loss_l,loss_c

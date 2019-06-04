from __future__ import print_function
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
import argparse
import numpy as np
from torch.autograd import Variable
import torch.utils.data as data
from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, COCODetection, VOCDetection, detection_collate, BaseTransform, preproc, preproc_mixup
from layers.modules import MultiBoxLoss
from layers.functions import PriorBox,Detect
import time
import math
from val import val_net

parser = argparse.ArgumentParser(description='SSD Training')
parser.add_argument('-v', '--version', default='SSD', help='version.')
parser.add_argument('-s', '--size', default='300', help='300 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC', help='VOC or COCO dataset')
parser.add_argument('--basenet', default='vgg16_bn.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8, type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=bool, help='Use cuda to train model')
parser.add_argument('--ngpu', default=1, type=int, help='gpus')
parser.add_argument('--lr', '--learning-rate', default=4e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument( '--resume_net', default=None, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('-max','--max_epoch', default=250, type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True, type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='./weights/', help='Location to save checkpoint models')
parser.add_argument('--save_val_folder', default='eval/', type=str, help='Dir to save results')
parser.add_argument('-wu','--warm_epoch', default='5', type=int, help='warm up')
parser.add_argument('-ls','--lr_schedule', default='cos', type=str, help='lr schedule: step;cos;htd')
parser.add_argument('--norm', default="BN", type=str, help='L2Norm/BN/GN for normalization')
parser.add_argument('-bd','--bias_decay', default=True, type=bool, help='BN/GN and bias for weight decay')
parser.add_argument('--label_smooth', default=False, type=bool,
                    help='Label Smooth for cls task, default label_pos=0.9.Please refer layers/modules/multibox_loss.py')
parser.add_argument('--balance_l1', default=False, type=bool, help='Balanced for SmoothL1, refer to Libra R-CNN')
parser.add_argument('--random_erasing', default=True, type=bool, help='Random Erasing for Data Augmentation')
parser.add_argument('--focal_loss', default=False, type=bool, help='Focal Loss')
parser.add_argument('--alpha', default=0, type=float, help='Mixup for SSD, if alpha is zero, not use Mixup')
parser.add_argument('--giou', default=False, type=bool, help='GIOU for reg loss')
parser.add_argument('--vgg_bn', default=True, type=bool, help='Use VGG16_BN as backbone for training')
args = parser.parse_args()


if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if args.dataset == 'VOC':
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    cfg = (VOC_300, VOC_512)[args.size == '512']
else:
    train_sets = [('2014', 'train'),('2014', 'valminusminival')]
    cfg = (COCO_300, COCO_512)[args.size == '512']

if args.version == 'SSD':
    from models.SSD import build_net
else:
    print('Unkown version!')

img_dim = (300,512)[args.size=='512']
rgb_means = (104, 117, 123)
p = 0.6
num_classes = (21, 81)[args.dataset == 'COCO']
batch_size = args.batch_size
weight_decay = args.weight_decay
gamma = args.gamma
momentum = args.momentum

net = build_net(img_dim, num_classes,args.norm,args.vgg_bn)
print(net)
if not args.resume_net:
    base_weights = torch.load(args.basenet)
    print('Loading base network...')
    if args.vgg_bn:
        net.base[:-5].load_state_dict(base_weights)
    else:
        net.base.load_state_dict(base_weights)

    def weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                if 'conv' in key:
                    init.kaiming_normal_(m.state_dict()[key], mode='fan_out',nonlinearity='relu')
                if 'bn' in key:
                    m.state_dict()[key][...] = 1
                if 'gn' in key:
                    m.state_dict()[key][...] = 1

            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    def head_weights_init(m):
        for key in m.state_dict():
            if key.split('.')[-1] == 'weight':
                init.xavier_uniform_(m.state_dict()[key])
            elif key.split('.')[-1] == 'bias':
                m.state_dict()[key][...] = 0

    print('Initializing weights...')
    # initialize newly added layers' weights with kaiming_normal method
    if args.vgg_bn:
        net.base[-5:].apply(weights_init)
    net.extras.apply(weights_init)

else:
    print('Loading resume network')
    state_dict = torch.load(args.resume_net)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict

    # multi-GPU
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:]  # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.cuda:
    net.cuda()
    cudnn.benchmark = True

if not args.bias_decay: # BN/GN and bias don't use weight decay
    spe_params = []
    conv_params = []
    for k, v in net.named_parameters():
        if 'bn' in k or 'bias' in k:
            spe_params.append(v)
        else:
            conv_params.append(v)
    params_group = [{'params': spe_params, 'weight_decay': 0.0}, {'params': conv_params}]
    optimizer = optim.SGD(params_group, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
else:
    optimizer = optim.SGD(net.parameters(), lr=args.lr,momentum=args.momentum, weight_decay=args.weight_decay)

criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False,label_smmooth=args.label_smooth,balance_l1=args.balance_l1,
                         focal_loss=args.focal_loss,giou=args.giou)

priorbox = PriorBox(cfg)
with torch.no_grad():
    priors = priorbox.forward()
    if args.cuda:
        priors = priors.cuda()

def get_features_hook(self,input,output):
    print('~'*10)
    print('features:')
    print('input:',input[0][0,0])
    print('output:',output[0,0])

def get_grads_hook(self,input_grad, output_grad):
    print('~'*10)
    print('grad:')
    print('grad_in:',input_grad[0][0,0])
    print('grad_out',output_grad[0][0,0])

def train():
    net.train()
    # loss counters
    loc_loss = 0  # epoch
    conf_loss = 0
    epoch = 0 + args.resume_epoch
    print('Loading Dataset...')

    if args.dataset == 'VOC':
        if args.alpha - 0.0 > 1e-5:
            dataset = VOCDetection(VOCroot, train_sets, preproc_mixup(img_dim, rgb_means, p), AnnotationTransform(), random_erasing=args.random_erasing,
                                   mixup_alpha=args.alpha)
        else:
            dataset = VOCDetection(VOCroot, train_sets, preproc(img_dim, rgb_means, p), AnnotationTransform(), random_erasing=args.random_erasing)
    elif args.dataset == 'COCO':
        dataset = COCODetection(COCOroot, train_sets, preproc(img_dim, rgb_means, p))
    else:
        print('Only VOC and COCO are supported now!')
        return

    epoch_size = len(dataset) // args.batch_size
    max_iter = args.max_epoch * epoch_size

    stepvalues_VOC = (150 * epoch_size, 200 * epoch_size, 250 * epoch_size)
    stepvalues_COCO = (100 * epoch_size, 135 * epoch_size, 170 * epoch_size)
    stepvalues = (stepvalues_VOC,stepvalues_COCO)[args.dataset=='COCO']
    print('Training',args.version, 'on', dataset.name)
    step_index = 0

    if args.resume_epoch > 0:
        start_iter = args.resume_epoch * epoch_size
        for sv in stepvalues:
            if start_iter>sv:
                step_index+=1
                continue
            else:
                break
    else:
        start_iter = 0

    lr = args.lr
    avg_loss_list = []
    flag = True
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            # create batch iterator
            batch_iterator = iter(data.DataLoader(dataset, batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=detection_collate))
            avg_loss = (loc_loss+conf_loss)/epoch_size
            avg_loss_list.append(avg_loss)
            print("avg_loss_list:")
            if len(avg_loss_list)<=5:
                print (avg_loss_list)
            else:
                print(avg_loss_list[-5:])
            loc_loss = 0
            conf_loss = 0
            if (epoch<=150 and epoch%10==0) or (150< epoch< 200 and epoch%5==0) or (epoch>200):
                torch.save(net.state_dict(), args.save_folder+args.version+'_'+args.dataset + '_epoches_'+ repr(epoch) + '.pth')
                if (epoch!=args.resume_epoch):
                #if(epoch):
                    ValNet = build_net(img_dim, num_classes, args.norm, args.vgg_bn)
                    val_state_dict = torch.load(args.save_folder + args.version + '_' + args.dataset + '_epoches_' + repr(epoch) + '.pth')
                    from collections import OrderedDict
                    new_state_dict = OrderedDict()
                    for k, v in val_state_dict.items():
                        head = k[:7]
                        if head == 'module.':
                            name = k[7:]
                        else:
                            name = k
                        new_state_dict[name] = v
                    ValNet.load_state_dict(new_state_dict)
                    ValNet.eval()
                    print('Finished loading ' + args.version + '_' + args.dataset + '_epoches_' + repr(epoch) + '.pth model!')
                    if args.dataset == 'VOC':
                        testset = VOCDetection(VOCroot, [('2007', 'test')], None, AnnotationTransform())
                    elif args.dataset == 'COCO':
                        testset = COCODetection(COCOroot, [('2014', 'minival')], None)
                    if args.cuda:
                        ValNet = ValNet.cuda()
                        cudnn.benchmark = True
                    else:
                        ValNet = ValNet.cpu()
                    top_k = 200
                    detector = Detect(num_classes, 0, cfg, GIOU=args.giou)
                    save_val_folder = os.path.join(args.save_val_folder, args.dataset)
                    val_transform = BaseTransform(ValNet.size, rgb_means, (2, 0, 1))
                    val_net(priors, save_val_folder, testset, num_classes, ValNet, detector, val_transform, top_k, 0.01,
                            args.cuda,args.vgg_bn)
            epoch += 1

        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        images, targets = next(batch_iterator)

        # no mixup
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(anno.cuda()) for anno in targets]
        else:
            images = Variable(images)
            targets = [Variable(anno) for anno in targets]

        # fh = net.base[22].register_forward_hook(get_features_hook)
        # bh = net.base[22].register_backward_hook(get_grads_hook)
        out = net(images,vgg_bn=args.vgg_bn)
        optimizer.zero_grad()
        loss_l, loss_c, = criterion(out, priors, targets)
        loss = loss_l + loss_c
        loss.backward()
        # fh.remove()
        # bh.remove()

        optimizer.step()
        t1 = time.time()
        loc_loss += loss_l.item()
        conf_loss += loss_c.item()
        load_t1 = time.time()
        if iteration % 10 == 0:
            print('Epoch:' + repr(epoch) + ' || epochiter: ' + repr(iteration % epoch_size) + '/' + repr(epoch_size)
                  + '|| Totel iter ' + repr(iteration) + ' || L: %.4f C: %.4f S: %.4f||' % (loss_l.item(),loss_c.item(),loss_l.item()+loss_c.item()) +
                'Batch time: %.4f ||' % (load_t1 - load_t0) + 'LR: %.7f' % (lr))

    torch.save(net.state_dict(), args.save_folder + 'Final_' + args.version +'_' + args.dataset+ '.pth')


def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size,lr_schedule=args.lr_schedule):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch <= args.warm_epoch:
        lr = 1e-6 + (args.lr - 1e-6) * iteration / (epoch_size * args.warm_epoch)
    else:
        if lr_schedule == 'step':
            lr = args.lr * (gamma ** (step_index))
        elif lr_schedule == 'cos':
            lr = 1e-6 + (args.lr - 1e-6) * 0.5 * (1 + math.cos(
                (iteration - args.warm_epoch * epoch_size) * math.pi /((args.max_epoch - args.warm_epoch) * epoch_size)))
        elif lr_schedule == 'htd':
            l,u = -6,3
            lr = 1e-6 + (args.lr - 1e-6) * 0.5 * (1 - math.tanh(l + (u - l) *
                ((iteration - args.warm_epoch * epoch_size) /(args.max_epoch - args.warm_epoch) /epoch_size)))
        else:
            print ('Unknown the lr schedule type!')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':
    train()


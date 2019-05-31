"""Data augmentation functionality. Passed as callable transformations to
Dataset classes.

The data augmentation procedures were interpreted from @weiliu89's SSD paper
http://arxiv.org/abs/1512.02325
"""

import torch
from torchvision import transforms
import cv2
import numpy as np
import random
import math
from utils.box_utils import matrix_iou
# import torch_transforms

def _crop(image, boxes, labels):
    height, width, _ = image.shape

    if len(boxes)== 0:
        return image, boxes, labels

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3,1.)
            min_ratio = max(0.5, scale*scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)


            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])
            
            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t,labels_t

def _crop_mixup(image, boxes, labels, weights):
    height, width, _ = image.shape

    if len(boxes) == 0:
        return image, boxes, labels, weights

    while True:
        mode = random.choice((
            None,
            (0.1, None),
            (0.3, None),
            (0.5, None),
            (0.7, None),
            (0.9, None),
            (None, None),
        ))

        if mode is None:
            return image, boxes, labels, weights

        min_iou, max_iou = mode
        if min_iou is None:
            min_iou = float('-inf')
        if max_iou is None:
            max_iou = float('inf')

        for _ in range(50):
            scale = random.uniform(0.3, 1.)
            min_ratio = max(0.5, scale * scale)
            max_ratio = min(2, 1. / scale / scale)
            ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
            w = int(scale * ratio * width)
            h = int((scale / ratio) * height)

            l = random.randrange(width - w)
            t = random.randrange(height - h)
            roi = np.array((l, t, l + w, t + h))

            iou = matrix_iou(boxes, roi[np.newaxis])

            if not (min_iou <= iou.min() and iou.max() <= max_iou):
                continue

            image_t = image[roi[1]:roi[3], roi[0]:roi[2]]

            centers = (boxes[:, :2] + boxes[:, 2:]) / 2
            mask = np.logical_and(roi[:2] < centers, centers < roi[2:]).all(axis=1)
            boxes_t = boxes[mask].copy()
            labels_t = labels[mask].copy()
            weights_t = weights[mask].copy()
            if len(boxes_t) == 0:
                continue

            boxes_t[:, :2] = np.maximum(boxes_t[:, :2], roi[:2])
            boxes_t[:, :2] -= roi[:2]
            boxes_t[:, 2:] = np.minimum(boxes_t[:, 2:], roi[2:])
            boxes_t[:, 2:] -= roi[:2]

            return image_t, boxes_t, labels_t, weights_t


def _distort(image):
    def _convert(image, alpha=1, beta=0):
        tmp = image.astype(float) * alpha + beta
        tmp[tmp < 0] = 0
        tmp[tmp > 255] = 255
        image[:] = tmp

    image = image.copy()

    if random.randrange(2):
        _convert(image, beta=random.uniform(-32, 32))

    if random.randrange(2):
        _convert(image, alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    if random.randrange(2):
        tmp = image[:, :, 0].astype(int) + random.randint(-18, 18)
        tmp %= 180
        image[:, :, 0] = tmp

    if random.randrange(2):
        _convert(image[:, :, 1], alpha=random.uniform(0.5, 1.5))

    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def _expand(image, boxes,fill, p):
    if random.random() > p:
        return image, boxes

    height, width, depth = image.shape
    for _ in range(50):
        scale = random.uniform(1,4)

        min_ratio = max(0.5, 1./scale/scale)
        max_ratio = min(2, scale*scale)
        ratio = math.sqrt(random.uniform(min_ratio, max_ratio))
        ws = scale*ratio
        hs = scale/ratio
        if ws < 1 or hs < 1:
            continue
        w = int(ws * width)
        h = int(hs * height)

        left = random.randint(0, w - width)
        top = random.randint(0, h - height)

        boxes_t = boxes.copy()
        boxes_t[:, :2] += (left, top)
        boxes_t[:, 2:] += (left, top)


        expand_image = np.empty(
            (h, w, depth),
            dtype=image.dtype)
        expand_image[:, :] = fill
        expand_image[top:top + height, left:left + width] = image
        image = expand_image

        return image, boxes_t


def _mirror(image, boxes):
    _, width, _ = image.shape
    if random.randrange(2):
        image = image[:, ::-1]
        boxes = boxes.copy()
        boxes[:, 0::2] = width - boxes[:, 2::-2]
    return image, boxes


def preproc_for_test(image, insize, mean):
    interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
    interp_method = interp_methods[random.randrange(5)]
    image = cv2.resize(image, (insize, insize),interpolation=interp_method)
    image = image.astype(np.float32)
    image -= mean
    return image.transpose(2, 0, 1)

def _random_erasing(image, boxes, means, p=0.6, sl=0.02, sh=0.2, r1=0.3):
    if random.uniform(0, 1) > p:
        return image
    area_boxes = (boxes[:,2]-boxes[:,0])*(boxes[:,3]-boxes[:,1])
    m = [2*j for j in means] # Please ensure the value of 2*mean is less to 255.
    for i in range(len(boxes)):
        for _ in range(50):
            area = random.uniform(sl,sh) * area_boxes[i]
            aspect_ratio = random.uniform(r1, 1.0 / r1)

            h = int(round(math.sqrt(area * aspect_ratio)))
            w = int(round(math.sqrt(area / aspect_ratio)))

            boxes_w = boxes[i,2] - boxes[i,0]
            boxes_h = boxes[i,3] - boxes[i,1]

            if w < boxes_w and h < boxes_h:
                x1 = int(random.randint(0,(boxes_w - w)) + boxes[i,0])
                y1 = int(random.randint(0,(boxes_h - h)) + boxes[i,1])
                image[y1:y1 + h, x1:x1 + w, :] = m
                break

    for j in range(50):
        area = random.uniform(sl, sh) * image.shape[0] * image.shape[1]
        aspect_ratio = random.uniform(r1, 1.0 / r1)
        h = int(round(math.sqrt(area * aspect_ratio)))
        w = int(round(math.sqrt(area / aspect_ratio)))

        if w < image.shape[1] and h < image.shape[0]:
            x1 = int(random.randint(0, (image.shape[1] - w)))
            y1 = int(random.randint(0, (image.shape[0] - h)))
            img_crop = np.array((x1,y1,x1+w,y1+h))

            ios = matrix_iou(boxes, img_crop[np.newaxis], erasing=True)
            if ios.max() < 0.2:
                image[y1:y1 + h, x1:x1 + w, :] = m
                break
    # cv2.imshow('eras.jpg',image)
    # cv2.waitKey()
    # exit()
    return image


class preproc(object):

    def __init__(self, resize, rgb_means, p):
        self.means = rgb_means
        self.resize = resize
        self.p = p

    def __call__(self, image, targets, random_erasing):
        boxes = targets[:,:-1].copy()
        labels = targets[:,-1].copy()
        if len(boxes) == 0:
            #boxes = np.empty((0, 4))
            targets = np.zeros((1,5))
            image = preproc_for_test(image, self.resize, self.means)
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:,:-1]
        labels_o = targets_o[:,-1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o,1)
        targets_o = np.hstack((boxes_o,labels_o))

        image_t, boxes, labels = _crop(image, boxes, labels)
        image_t = _distort(image_t)
        if random_erasing:
            image_t = _random_erasing(image_t, boxes, self.means)
        image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        image_t, boxes = _mirror(image_t, boxes)

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b= np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()

        if len(boxes_t)==0:
            image = preproc_for_test(image_o, self.resize, self.means)
            return torch.from_numpy(image),targets_o

        labels_t = np.expand_dims(labels_t,1)
        targets_t = np.hstack((boxes_t,labels_t))

        return torch.from_numpy(image_t), targets_t

class preproc_mixup(object):

    def __init__(self, resize, rgb_means, p):
        self.means = rgb_means
        self.resize = resize
        self.p = p

    def __call__(self, image, targets, random_erasing):
        boxes = targets[:,:-2].copy()
        labels = targets[:,-2].copy()
        weights = targets[:,-1].copy()
        if len(boxes) == 0:
            #boxes = np.empty((0, 4))
            targets = np.zeros((1,6))
            image = preproc_for_test(image, self.resize, self.means)
            return torch.from_numpy(image), targets

        image_o = image.copy()
        targets_o = targets.copy()
        height_o, width_o, _ = image_o.shape
        boxes_o = targets_o[:,:-2]
        labels_o = targets_o[:,-2]
        weights_o = targets_o[:, -1]
        boxes_o[:, 0::2] /= width_o
        boxes_o[:, 1::2] /= height_o
        labels_o = np.expand_dims(labels_o,1)
        weights_o = np.expand_dims(weights_o, 1)
        targets_o = np.hstack((boxes_o,labels_o,weights_o))

        image_t, boxes, labels, weights = _crop_mixup(image, boxes, labels, weights)
        image_t = _distort(image_t)
        if random_erasing:
            image_t = _random_erasing(image_t, boxes, self.means)
        image_t, boxes = _expand(image_t, boxes, self.means, self.p)
        image_t, boxes = _mirror(image_t, boxes)

        height, width, _ = image_t.shape
        image_t = preproc_for_test(image_t, self.resize, self.means)
        boxes = boxes.copy()
        boxes[:, 0::2] /= width
        boxes[:, 1::2] /= height
        b_w = (boxes[:, 2] - boxes[:, 0])*1.
        b_h = (boxes[:, 3] - boxes[:, 1])*1.
        mask_b= np.minimum(b_w, b_h) > 0.01
        boxes_t = boxes[mask_b]
        labels_t = labels[mask_b].copy()
        weights_t = weights[mask_b].copy()

        if len(boxes_t)==0:
            image = preproc_for_test(image_o, self.resize, self.means)
            return torch.from_numpy(image),targets_o

        labels_t = np.expand_dims(labels_t,1)
        weights_t = np.expand_dims(weights_t,1)
        targets_t = np.hstack((boxes_t,labels_t,weights_t))



        return torch.from_numpy(image_t), targets_t

class BaseTransform(object):
    """Defines the transformations that should be applied to test PIL image
        for input into the network

    dimension -> tensorize -> color adj

    Arguments:
        resize (int): input dimension to SSD
        rgb_means ((int,int,int)): average RGB of the dataset
            (104,117,123)
        swap ((int,int,int)): final order of channels
    Returns:
        transform (transform) : callable transform to be applied to test/val
        data
    """
    def __init__(self, resize, rgb_means, swap=(2, 0, 1)):
        self.means = rgb_means
        self.resize = resize
        self.swap = swap

    # assume input is cv2 img for now
    def __call__(self, img):

        interp_methods = [cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_NEAREST, cv2.INTER_LANCZOS4]
        interp_method = interp_methods[0]
        img = cv2.resize(np.array(img), (self.resize,
                                         self.resize),interpolation = interp_method).astype(np.float32)
        img -= self.means
        img = img.transpose(self.swap)
        return torch.from_numpy(img)

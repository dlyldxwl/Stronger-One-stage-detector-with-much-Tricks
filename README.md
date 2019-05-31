# Stronger One-stage detector with much Tricks

This repo was inspired by the paper [Bag of Freebies for Training Object Detection Neural Networks](https://arxiv.org/pdf/1902.04103).

I would test popular training tricks as many as I can for improving one-stage detector accuarcy, feel free to leave a comment or email me about the tricks you want me to test ([yhao.chen0617@gmail.com](yhao.chen0617@gmail.com)).

**Traing Data** :  VOC0712 trainval
**Test data** :  VOC07 test
**GPU** :  TITAN X(pascal)
**Framework** :  Pytorch 0.4

Network | mAP | FPS | Parameter
--|:--:|:--:|:--:
SSD 300| 79.62 | ~100| -
YOLOV3 544| - | - | - 

**Note**:

- [ ] Stronger YOLOv3 with much tricks will be released soon.
- [ ] This repo does not use multi-scale train because of the limitation of GPU memory (I only have one card), which is extremely beneficial to detector.

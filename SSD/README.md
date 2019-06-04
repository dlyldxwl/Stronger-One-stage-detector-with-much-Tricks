# Stronger SSD with much Tricks
## Tricks
This repo was mainly used the following tricks.

Trick | Reference paper
--|:--:
Warm up | -
Cos lr | -
Htd lr | -
Batch Normalization | -
Group Normalization | -
No bais decay | -
Label smooth | -
Mixup | -
Random erasing | -
Balance Smoothl1 | -
Focal loss | -
GIOU | -
Octconv | -


## Result
Pretrained model is VGG-16 (atrous). The size of all models is 300&times;300.

**SSD equips much data augmentation operations, which leads miuxp, label smooth and some data augmentation methods or regularization don't work.** 

## Note

- [ ] 80.58 is not the final resualt. The experiment of SSD300 with Focal loss, GIoU and Octconv is still going on. 

- [ ] BN can merge into convolution layer, thus it will not increase any inference time and parameters. The merge code will be pubilc soon.

- [ ] Multi-Scale traing with SSD 300 will acquire a significant gain, which will be released when I go to the internship (about one month later), because I only have one GPU now.

# A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection
By Xiaolong Wang, Abhinav Shrivastava, and Abhinav Gupta

### Introduction

This is a Caffe based version of A-Fast-RCNN ([arxiv_link](https://arxiv.org/pdf/1704.03414.pdf)). Although we originally implement it on torch, this Caffe re-implementation is much simpler, faster and easier to use.

We release the code for training A-Fast-RCNN with Adversarial Spatial Dropout Network.


### License

This code is released under the MIT License (refer to the LICENSE file for details).

### Citing

If you find this useful in your research, please consider citing:

    @inproceedings{WangCVPR17afrcnn,
        Author = {Xiaolong Wang and Abhinav Shrivastava and Abhinav Gupta},
        Title = {A-Fast-RCNN: Hard Positive Generation via Adversary for Object Detection},
        Booktitle = {Conference on Computer Vision and Pattern Recognition ({CVPR})},
        Year = {2017}
    }

### Disclaimer

This implementation is built on a *fork* of the OHEM code ([here](https://github.com/abhi2610/ohem)), which in turn builds on the Faster R-CNN Python code ([here](https://github.com/rbgirshick/py-faster-rcnn)) and Fast R-CNN ([here](https://github.com/rbgirshick/fast-rcnn)). Please cite the appropriate papers depending on which part of the code and/or model you are using.

### Results
    | Approach                       | training data           | test data         | mAP
    | Fast R-CNN  (FRCN)             | VOC 07 trainval         | VOC 07 test       | 67.6
    | FRCN with adversary            | VOC 07 trainval         | VOC 07 test       | 70.8

**Note**: The reported results are based on the VGG16 network.



### Installation

Please follow the exact installation and download the VOC data as the Faster R-CNN Python code ([here](https://github.com/rbgirshick/py-faster-rcnn)).

### Usage

To run the code, one can simply do,
```Shell
./train.sh
```

It includes 3-stage of training:

```Shell
./experiments/scripts/fast_rcnn_std.sh  [GPU_ID]  VGG16 pascal_voc
```
which is used for training a standard Fast-RCNN for 10K iterations, you can download my [model](https://www.dropbox.com/s/ccs7lw3gydfzgvv/fast_rcnn_std_iter_10000.caffemodel?dl=0) and [logs](https://www.dropbox.com/s/hwbag60l1gmtxbb/fast_rcnn_std.txt.2017-04-08_16-53-59?dl=0) for this step.

```Shell
./experiments/scripts/fast_rcnn_adv_pretrain.sh  [GPU_ID]  VGG16 pascal_voc
```
which is a pre-training stage for the adversarial network, you can download my [model](https://www.dropbox.com/s/hvqpxn3bigarhdn/fast_rcnn_adv_pretrain_iter_25000.caffemodel?dl=0) and [logs](https://www.dropbox.com/s/i79j5hd0ee4ybke/fast_rcnn_adv_pretrain.txt.2017-04-08_19-39-49?dl=0) for this step.

```Shell
./copy_model.h
```
which is used to copy the weights of the above two models to initialize the joint model.

```Shell
./experiments/scripts/fast_rcnn_adv.sh  [GPU_ID]  VGG16 pascal_voc
```
which is joint training of the detector and the adversarial network, you can download my [model](https://www.dropbox.com/s/5wvxh8g5n3ewvp4/fast_rcnn_adv_iter_40000.caffemodel?dl=0) and [logs](https://www.dropbox.com/s/awrdrwyfthdgba5/fast_rcnn_adv.txt.2017-04-09_22-09-57?dl=0) for this step.

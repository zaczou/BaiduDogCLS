#!/bin/bash

# first fine tune the last layer only
export PATH=/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS/caffe/build/tools:$PATH
# caffe train --solver="model/VGG16/ft_last_layer.solver" --weights="model/VGG16/VGG_ILSVRC_16_layers.caffemodel" --gpu=6,7 2>&1 |tee log/crop_vgg16_last.log &
#caffe train --solver="model/VGG19/ft_last_layer.solver" --weights="model/VGG19/VGG_ILSVRC_19_layers.caffemodel" --gpu=4,5 2>&1 |tee log/vgg19_last.log &
# caffe train --solver="model/resnet50/ft_last_layer.solver" --weights="model/resnet50/ResNet-50-model.caffemodel" --gpu=3,2 2>&1 |tee log/resenet50_last.log
caffe train --solver="model/resnext50/ft_last_layer.solver" --weights="model/resnext50/resnext50.caffemodel" --gpu=4,5 2>&1 |tee log/resnext50_last.log

#!/bin/bash

# first fine all layers
export PATH=/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS/caffe/build/tools:$PATH


#caffe train --solver="model/VGG19/ft_all.solver" --weights="snapshot/VGG19/ft_last_layer_iter_60000.caffemodel" --gpu=4,5 2>&1 |tee log/vgg19_all.log
#caffe train --solver="model/VGG16/ft_all.solver" --weights="snapshot/VGG16/full/ft_last_layer_iter_60000.caffemodel" --gpu=2,3 2>&1 |tee log/vgg16_all.log &
#caffe train --solver="model/resnet50/ft_all.solver" --weights="snapshot/resnet50/ft_last_layer_iter_50000.caffemodel" --gpu=2,3 2>&1 |tee log/resnet50_all.log
#caffe train --solver="model/test/ft_all.solver" --weights="model/resnet50/ResNet-50-model.caffemodel" --gpu=2,3,4,5 2>&1 |tee log/test_all.log
caffe train --solver="model/resnet50/ft_all.solver" --weights="model/resnet50/ResNet-50-model.caffemodel" --gpu=0,1,2,3 2>&1 |tee log/resnet50_all.log
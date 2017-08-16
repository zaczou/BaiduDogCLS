#!/bin/bash

# first fine all layers

log_dir=/home/lab-xiong.jiangfeng/Projects/doge/BoostCNN/examples/compact_bilinear/
./build/tools/caffe train --solver="examples/compact_bilinear/ft_all.solver" --weights="examples/compact_bilinear/snapshot/ft_last_layer_iter_60000.caffemodel" --gpu=0 2>&1 | tee ${log_dir}ft_all.log
	
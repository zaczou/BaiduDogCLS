export PATH=/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS/caffe/build/tools:$PATH

mkdir model/snapshot/base
caffe train --solver=solver/solver.prototxt --gpu=5 2>&1 | tee log/base.log

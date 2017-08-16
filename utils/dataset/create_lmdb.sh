#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
#set caffe path
export PATH=/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS/caffe/build/tools:$PATH

set -e

EXAMPLE=dataset
DATA=dataset/train
#TOOLS=build/tools

# TRAIN_DATA_ROOT=dataset/train/crop_body/
# VAL_DATA_ROOT=dataset/train/crop_body/

TRAIN_DATA_ROOT=dataset/train/IMG/
VAL_DATA_ROOT=dataset/train/IMG/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=false
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_lmdb.sh to the path" \
       "where the training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_lmdb.sh to the path" \
       "where the validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=true \
    --encoded=false \
    $TRAIN_DATA_ROOT \
    $DATA/train_list.txt \
    $EXAMPLE/doge_train_keep_ratio_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=true \
    --encoded=false \
    $VAL_DATA_ROOT \
    $DATA/val_list.txt \
    $EXAMPLE/doge_val_keep_ratio_lmdb

echo "Creating trainval lmdb..."
GLOG_logtostderr=1 convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle=true \
    --encoded=true \
    $VAL_DATA_ROOT \
    $DATA/trainval_list.txt \
    $EXAMPLE/doge_trainval_keep_ratio_lmdb

echo "Done."

#!/bin/bash

TOOLS='.'
LOGDIR='../'
num_interval=$2
log_file=$1
cp $LOGDIR/$log_file $log_file
$TOOLS/parse_log.sh $log_file

#plot_training_log Usage:
#    ./plot_training_log.py chart_type[0-7] /where/to/save.png /path/to/first.log ...
#Notes:
#    1. Supporting multiple logs.
#    2. Log file name must end with the lower-cased ".log".
#Supported chart types:
#    0: Test accuracy  vs. Iters
#    1: Test accuracy  vs. Seconds
#    2: Test loss  vs. Iters
#    3: Test loss  vs. Seconds
#    4: Train learning rate  vs. Iters
#    5: Train learning rate  vs. Seconds
#    6: Train loss  vs. Iters
#    7: Train loss  vs. Seconds
$TOOLS/plot_training_log.py.example 4 lr.png $num_interval $log_file
$TOOLS/plot_training_log.py.example 6 train_loss.png $num_interval $log_file
$TOOLS/plot_training_log.py.example 0 test_acc.png $num_interval $log_file
$TOOLS/plot_training_log.py.example 2 test_loss.png $num_interval $log_file



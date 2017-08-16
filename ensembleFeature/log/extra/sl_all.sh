#!/bin/bash


TOOLS='.'
LOGDIRS=('base')


num_interval=$1
log_list=''
for log in ${LOGDIRS[@]}
do
	log_file=$log.log
	cp ../$log_file $log.log
	$TOOLS/parse_log.sh $log.log
	log_list=${log_list}${log}'.log '
done
echo $log_list

# $TOOLS/plot_training_log.py.example 4 lr.png $num_interval $log_list
# $TOOLS/plot_training_log.py.example 6 train_loss.png $num_interval $log_list
# $TOOLS/plot_training_log.py.example 0 test_acc.png $num_interval $log_list
# $TOOLS/plot_training_log.py.example 2 test_loss.png $num_interval $log_list

$TOOLS/plot_training_log.py.example 4 lr.png $log_list
$TOOLS/plot_training_log.py.example 6 train_loss.png $log_list
$TOOLS/plot_training_log.py.example 0 test_acc.png $log_list
$TOOLS/plot_training_log.py.example 2 test_loss.png $log_list

rm *.log
#rm *.test
rm *.train
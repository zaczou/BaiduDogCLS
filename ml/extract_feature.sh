#!/usr/bin/env sh


export PATH=/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS/caffe/build/tools:$PATH
#uncomment to generate test file list
#Extract feature of test_file
# test_source_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/test-01"
# find ${test_source_dir}/crop_image -type f -exec echo {} \; > ${test_source_dir}/temp.txt
# sed "s/$/ 0/" ${test_source_dir}/temp.txt >${test_source_dir}/test-01_crop_list.txt
root_dir='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/IMG\/'

#RESNET50 299 feature
fea_name="pool5,label"
prefix="resnet50"
model_prototxt='model/resnet299.prototxt'
model_weight='/home/lab-xiong.jiangfeng/Projects/doge/backup/resnet50_iter_2600.caffemodel'

save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_train"
save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label"
source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/train_list.txt'
sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir},${save_dir_label} 15009 lmdb GPU 0

save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_val"
save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label_val"
source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/val_list.txt'
sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir},${save_dir_label} 3752 lmdb GPU 0

##Test 01
fea_name="pool5"
test_save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/test-01/feat/${prefix}"
test_source_file="\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/test-01\/test-01_list.txt"
test_root_dir=""
sed "s/source:.*/source: \"${test_source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
sed "s/root_folder:.*/root_folder: \"${test_root_dir}\"/" model/net_temp.prototxt > model/net.prototxt
extract_features ${model_weight} model/net.prototxt ${fea_name} ${test_save_dir} 10593 lmdb GPU 0


#RENEXT 224
# fea_name="pool_ave,label"
# prefix="resnext224"
# model_prototxt='model/resnext224.prototxt'
# model_weight='/home/lab-xiong.jiangfeng/Projects/doge/backup/resnext224_iter_1600.caffemodel'

# save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_train"
# save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label"
# source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/train_list.txt'
# sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir},${save_dir_label} 15009 lmdb GPU 0

# save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_val"
# save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label_val"
# source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/val_list.txt'
# sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir},${save_dir_label} 3752 lmdb GPU 0

# ##Test 01
# fea_name="pool_ave"
# test_save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/test-01/feat/${prefix}"
# test_source_file="\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/test-01\/test-01_list.txt"
# test_root_dir=""
# sed "s/source:.*/source: \"${test_source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${test_root_dir}\"/" model/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${test_save_dir} 10593 lmdb GPU 0



# #RESNEXT 299 feature
# fea_name="pool_ave,label"
# prefix="resnext299"
# model_prototxt='model/resnext299.prototxt'
# model_weight='/home/lab-xiong.jiangfeng/Projects/doge/backup/resnext299_iter_1800.caffemodel'

# save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_train"
# save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label"
# source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/train_list.txt'
# sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir},${save_dir_label} 15009 lmdb GPU 0

# save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_val"
# save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label_val"
# source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/val_list.txt'
# sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir},${save_dir_label} 3752 lmdb GPU 0

# ##Test 01
# fea_name="pool_ave"
# test_save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/test-01/feat/${prefix}"
# test_source_file="\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/test-01\/test-01_list.txt"
# test_root_dir=""
# sed "s/source:.*/source: \"${test_source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${test_root_dir}\"/" model/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${test_save_dir} 10593 lmdb GPU 0



#Inceptionv4
# prefix="inception-v4"
# fea_name="pool_8x8_s1_drop"
# model_prototxt='model/inception-v4.prototxt'
# model_weight='/home/lab-xiong.jiangfeng/Projects/doge/backup/inception-v4_iter_7000.caffemodel'

# save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_train"
# #save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label"
# source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/train_list.txt'
# sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir} 15009 lmdb GPU 0

# save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_val"
# save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label_val"
# source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/val_list.txt'
# sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir},${save_dir_label} 3752 lmdb GPU 0

# ##Test 01
# fea_name="pool_8x8_s1_drop"
# test_save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/test-01/feat/${prefix}"
# test_source_file="\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/test-01\/test-01_list.txt"
# test_root_dir=""
# sed "s/source:.*/source: \"${test_source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${test_root_dir}\"/" model/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${test_save_dir} 10593 lmdb GPU 0



# #InceptionV4-crop
# prefix="inception-v4_crop"
# fea_name="pool_8x8_s1_drop"
# model_prototxt='model/inception-v4.prototxt'
# model_weight='/home/lab-xiong.jiangfeng/Projects/doge/backup/inception-v4_crop_iter_6400.caffemodel'

# save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_train"
# #save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label"
# source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/train_list.txt'
# sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir} 15009 lmdb GPU 0

# save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_val"
# save_dir_label="/home/lab-xiong.jiangfeng/Projects/doge/dataset/train/feat/${prefix}_label_val"
# source_file='\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/train\/val_list.txt'
# sed "s/source:.*/source: \"${source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${root_dir}\"/" model\/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${save_dir} 3752 lmdb GPU 0

# ##Test 01
# fea_name="pool_8x8_s1_drop"
# test_save_dir="/home/lab-xiong.jiangfeng/Projects/doge/dataset/test-01/feat/${prefix}"
# test_source_file="\/home\/lab-xiong.jiangfeng\/Projects\/doge\/dataset\/test-01\/test-01_list.txt"
# test_root_dir=""
# sed "s/source:.*/source: \"${test_source_file}\"/" ${model_prototxt} > model/net_temp.prototxt
# sed "s/root_folder:.*/root_folder: \"${test_root_dir}\"/" model/net_temp.prototxt > model/net.prototxt
# extract_features ${model_weight} model/net.prototxt ${fea_name} ${test_save_dir} 10593 lmdb GPU 0

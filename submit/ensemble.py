#load library
import os
import numpy as np
import sys
from utils import *
import scipy.io
import matplotlib.pyplot as plt
from skimage.transform import resize
Project_dir="/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS"
caffe_root=os.path.join(Project_dir,"caffe/build/install")
sys.path.insert(0,os.path.join(Project_dir,"caffe/build/install/python"))
import caffe

caffe.set_device(0)
caffe.set_mode_gpu()

# transformer = caffe.io.Transformer({'data': [1,3,224,224]})
# transformer.set_transpose('data', (2,0,1))
# transformer.set_mean('data', np.array([104,117,123]))
# transformer.set_channel_swap('data', (2,1,0))
# transformer.set_raw_scale('data',255)

# transformer_2 = caffe.io.Transformer({'data': [1,3,299,299]})
# transformer_2.set_transpose('data', (2,0,1))
# transformer_2.set_mean('data', np.array([104,117,123]))
# transformer_2.set_channel_swap('data', (2,1,0))
# transformer_2.set_raw_scale('data',255)

transformer_1 = caffe.io.Transformer({'data': [1,3,224,224]})
transformer_1.set_transpose('data', (2,0,1))
transformer_1.set_mean('data', np.array([128,128,128]))
transformer_1.set_channel_swap('data', (2,1,0))
transformer_1.set_raw_scale('data',255)
transformer_1.set_input_scale('data',1.0/128.0)

transformer_2 = caffe.io.Transformer({'data': [1,3,299,299]})
transformer_2.set_transpose('data', (2,0,1))
transformer_2.set_mean('data', np.array([128,128,128]))
transformer_2.set_channel_swap('data', (2,1,0))
transformer_2.set_raw_scale('data',255)
transformer_2.set_input_scale('data',1.0/128.0)


crop_num = 10
repeat_num = 1
models=[]
#train
# resnet50_center_loss = SingleModel(model_def='/home/lab-xiong.jiangfeng/Projects/doge/model/deploy/resnet50_deploy.prototxt',
#                                    weights='/home/lab-xiong.jiangfeng/Projects/doge/model/SnapShot/res50_center_loss_aug_0001/_iter_2600.caffemodel',
#                                    transformer=transformer_2,crop_num=crop_num,min_side=320,crop_size=299)
# #
# resnext50_center_loss = SingleModel(model_def=os.path.join(Project_dir,'finetune/model/deploy/resnext50_deploy.prototxt'),
#                                     weights=os.path.join(Project_dir,'finetune/model/snapshot/resnext224/_iter_4400.caffemodel'),#0.8113
#                                     transformer=transformer,crop_num=crop_num,min_side=256,crop_size=224)
#
# resnext50_center_loss_large_scale = SingleModel(model_def=os.path.join(Project_dir,'finetune/model/deploy/resnext50_deploy.prototxt'),
#                                      weights=os.path.join(Project_dir,'finetune/model/snapshot/resnext299/_iter_8400.caffemodel'),#0.8203
#                                     transformer=transformer_2,crop_num=crop_num,min_side=320,crop_size=299)
#
#
# inceptionv4_centerloss = SingleModel(model_def=os.path.join(Project_dir,'finetune/model/deploy/inception-v4_deploy.prototxt'),
#                                      weights=os.path.join(Project_dir,'finetune/model/snapshot/inception-v4_centerloss/_iter_5200.caffemodel'),#0.8299
#                                      transformer=transformer_3,crop_num=crop_num,min_side=320,crop_size=299)
#
# inceptionv4_centerloss_v2 = SingleModel(model_def=os.path.join(Project_dir,'finetune/model/deploy/inception-v4_deploy.prototxt'),
#                                      weights=os.path.join(Project_dir,'finetune/model/snapshot/inception-v4_centerloss_v2/_iter_5600.caffemodel'),#0.831023
#                                      transformer=transformer_2,crop_num=crop_num,min_side=320,crop_size=299)
#

#==peeudo_labeling


# resnext50_center_loss = SingleModel(model_def=os.path.join(Project_dir,'finetune/model/deploy/resnext50_deploy.prototxt'),
#                                     weights=os.path.join(Project_dir,'pseudo/model/snapshot/resnext224/_iter_10400.caffemodel'),#0.822495
#                                     transformer=transformer_1,crop_num=crop_num,min_side=256,crop_size=224)

# resnext50_center_loss_large_scale = SingleModel(model_def=os.path.join(Project_dir,'finetune/model/deploy/resnext50_deploy.prototxt'),
#                                      weights=os.path.join(Project_dir,'pseudo/finetune_models/resnext299_iter_400.caffemodel'),#0.829424
#                                     transformer=transformer_2,crop_num=crop_num,min_side=320,crop_size=299)
#
# inceptionv4_centerloss = SingleModel(model_def=os.path.join(Project_dir,'finetune/model/deploy/inception-v4_deploy.prototxt'),
#                                      weights=os.path.join(Project_dir,'pseudo/finetune_models/inception-v4_iter_2400.caffemodel'),#0.839
#                                      transformer=transformer_2,crop_num=crop_num,min_side=320,crop_size=299)

inceptionv4_crop = SingleModel(model_def=os.path.join(Project_dir,'finetune/model/deploy/inception-v4_crop_deploy.prototxt'),
                               weights=os.path.join(Project_dir,'finetune/model/finetune_models/inception-v4_crop_iter_7200.caffemodel'),
                               transformer=transformer_2,crop_num=crop_num,min_side=320,crop_size=299,use_crop_image=True)

#models.append(resnext50_center_loss_large_scale)
#models.append(inceptionv4_centerloss)
#models.append(inceptionv4_centerloss_v2)
models.append(inceptionv4_crop)


#models=[resnext50_center_loss,resnext50_center_loss_large_scale,inceptionv4_centerloss,inceptionv4_centerloss_v2]#val: 0.8406
model_weights=[1.0]*len(models)

def pseudo_labels():
    test_dir_1 = os.path.join(Project_dir, "dataset/test-01/image")
    test_dir_2=  os.path.join(Project_dir,"dataset/test-02/image")
    image_list_1 = os.listdir(test_dir_1)
    image_list_2 = os.listdir(test_dir_2)

    pseudo_labels_list=open(os.path.join(Project_dir,"dataset/pseudo_label_list_ensemble_all.txt"),'w')

    test1_size = len(image_list_1)
    test2_size= len(image_list_2)

    for index, image in enumerate(image_list_1):
        image_path = os.path.join(test_dir_1, image)
        vote_label = test_img(models, model_weights, [image_path],repeat_num=repeat_num)
        print >> pseudo_labels_list, "%s %d" % (image_path, vote_label[0])
        print "Processing %d/%d"%(index+1,test1_size),image_path,"psedu_label: ",vote_label[0]

    for index, image in enumerate(image_list_2):
        image_path = os.path.join(test_dir_2, image)
        vote_label = test_img(models, model_weights, [image_path],repeat_num=repeat_num)
        print >> pseudo_labels_list, "%s %d" % (image_path, vote_label[0])
        print "Processing %d/%d"%(index+1,test2_size), image_path, "psedu_label: ", vote_label[0]


def test_validation(batch_size=10):
    trainval_dir = os.path.join(Project_dir,"dataset/train/IMG")
    img_list, label_list = load_validation()
    acc_cls = 0
    paths=[]
    labels=[]
    num_correct = 0
    num_wrong = 0
    for index, img in enumerate(img_list):
        path = os.path.join(trainval_dir, img)
        paths.append(path)
        labels.append(int(label_list[index]))
        if (index+1)%batch_size==0:
            vote_labels= test_img(models,model_weights,paths,repeat_num=repeat_num)
            for batch_id,label in enumerate(vote_labels):
                #cacluate the accuracy
                if(label == labels[batch_id]):
                    acc_cls = acc_cls + 1
                print "pre_label: %d label: %d" % ( label,labels[batch_id])
            print "Processing %d/%d, acc=%.4f"%(index+1, len(img_list),acc_cls * 1.0 / (index + 1))
            paths=[]
            labels=[]
    if(len(paths)>0):
        vote_labels = test_img(models, paths)
        for batch_id, vote_label in enumerate(vote_labels):
            if (labels[batch_id] == vote_labels[batch_id]):
                acc_cls = acc_cls + 1
            print "Processing %d /%d: vote_label: %d label: %d" % (index+1, len(img_list), vote_label,labels[batch_id]),
        print ",acc=%.4f"%(acc_cls * 1.0 / (index + 1))

def submit():
    encode_dict = load_mapdict()
    #test_dir_1 = os.path.join(Project_dir,"dataset/test-01/image")
    test_dir_2=  os.path.join(Project_dir,"dataset/test-02/image")
    test_dir_2_crop = os.path.join(Project_dir,"dataset/test-02/crop_image")
    image_list = os.listdir(test_dir_2)
    image_list.sort()
    submit_file = open(os.path.join(Project_dir,"submit/submit-0822-crop-single.txt"), 'w')
    for index, image in enumerate(image_list):
        vote_label = test_img(models, model_weights,
                              [os.path.join(test_dir_2, image)],
                              [os.path.join(test_dir_2_crop,image)],
                              repeat_num=repeat_num)
        de_label = decode(encode_dict, str(vote_label[0]))
        print "Processing %d/%d" % (index, len(image_list)), image[0:-4], vote_label[0], "encode:%s,decode:%s" % (
        str(vote_label[0]), de_label)
        print >> submit_file, "%s\t%s" % (de_label, image[0:-4])
    submit_file.close()


#pseudo_labels()
#test_validation(1)#only support batch_size=1 now , since we need to do repeat due to lacking of memory
submit()


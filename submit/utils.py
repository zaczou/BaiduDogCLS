import numpy as np
import sys
import os
import time
Project_dir="/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS"
caffe_root=os.path.join(Project_dir,"caffe/build/install")
sys.path.insert(0,os.path.join(Project_dir,"caffe/build/install/python"))
import caffe
from skimage.transform import resize
caffe.set_device(5)
caffe.set_mode_gpu()



def softmax(x):
    return np.exp(x)/np.sum(np.exp(x),axis=0)


def decode(encode_dict,label):
    return encode_dict[label]


def encode(encode_dict,label):
    return encode_dict.keys()[encode_dict.values().index(label)]


def load_mapdict():
    encode_dict_file=open(os.path.join(Project_dir,"dataset/train/map_dict"),"r")
    encode_dict={}
    line = encode_dict_file.readline()
    while (line):
        line = line.strip()
        cell = line.split(' ')
        encode_dict[cell[0]]=cell[1]
        line = encode_dict_file.readline()

    encode_dict_file.close()
    return encode_dict


class SingleModel(object):
    def __init__(self,model_def,weights,transformer,crop_num,min_side,crop_size,use_crop_image=False):
        self.net = caffe.Net(model_def,weights,caffe.TEST)
        self.transformer = transformer
        self.crop_num=crop_num
        self.min_side = min_side
        self.crop_size = crop_size
        self.use_crop_image = use_crop_image
    def preprocess(self,images,crop_images=None):

        num_images = len(images)
        inputs=np.zeros([num_images*self.crop_num,self.crop_size,self.crop_size,3])
        input_crops = np.zeros([num_images*self.crop_num,self.crop_size,self.crop_size,3])
        trans_shape = [self.crop_num * num_images, 3, self.crop_size, self.crop_size]
        self.transformed_image = np.zeros(trans_shape)
        self.transformed_crop_image = np.zeros(trans_shape)

        for i in range(num_images):
            resize_image = self.keep_ratio_resize(images[i],self.min_side)
            if (self.crop_num > 1):
                inputs[i * self.crop_num:(i + 1) * self.crop_num] = caffe.io.oversample([resize_image], tuple([self.crop_size, self.crop_size]))[:self.crop_num]
            else:# center crop
                center = np.array(resize_image.shape[:2]) / 2.0
                crop = np.tile(center, (1, 2))[0] + np.concatenate(
                    [-np.array([self.crop_size, self.crop_size]) / 2.0, np.array([self.crop_size, self.crop_size]) / 2.0])
                crop = crop.astype(int)
                inputs[i * self.crop_num:(i + 1) * self.crop_num] = resize_image[crop[0]:crop[2], crop[1]:crop[3], :]
            if(self.use_crop_image):
                #default crop_num>0(TODO)
                resize_crop_image = resize(crop_images[i],(self.crop_size,self.crop_size))
                input_crops[i * self.crop_num:(i + 1) * self.crop_num] = caffe.io.oversample([resize_crop_image], tuple([self.crop_size, self.crop_size]))[:self.crop_num]

        for img_index in range(self.crop_num * num_images):
            self.transformed_image[img_index, :] = self.transformer.preprocess('data', inputs[img_index])
            if(self.use_crop_image):
                self.transformed_crop_image[img_index, :] = self.transformer.preprocess('data', input_crops[img_index])

    def forward(self,images,crop_images=None,prob_name="prob",repeat_num=1):
        self.preprocess(images,crop_images)
        mini_batchsize = self.crop_num / repeat_num
        score_means = []
        labels = []
        for n_rep in range(repeat_num):
            if(self.use_crop_image):
                self.net.blobs['raw_data'].reshape(mini_batchsize, 3, self.crop_size, self.crop_size)
                self.net.blobs['raw_data'].data[...] = self.transformed_image[n_rep*mini_batchsize:(n_rep+1)*mini_batchsize]
                self.net.blobs['crop_data'].reshape(mini_batchsize, 3, self.crop_size, self.crop_size)
                self.net.blobs['crop_data'].data[...] = self.transformed_crop_image[n_rep*mini_batchsize:(n_rep+1)*mini_batchsize]
            else:
                self.net.blobs['data'].reshape(mini_batchsize, 3, self.crop_size, self.crop_size)
                self.net.blobs['data'].data[...] = self.transformed_image[n_rep * mini_batchsize:(n_rep + 1) * mini_batchsize]

            self.net.forward()
            scores = self.net.blobs[prob_name].data
            score_means.append(np.mean(scores, axis=0))
        score = np.mean(np.asarray(score_means),axis=0)
        label = np.argmax(score)
        return label,score

    def keep_ratio_resize(self,image,min_side=256):
        if (image.shape[0] > image.shape[1]):
            resize_shape_1 = min_side
            resize_shape_0 = np.ceil((image.shape[0] * 1.0 / image.shape[1]) * min_side)
            return resize(image, (resize_shape_0, resize_shape_1))
        else:
            resize_shape_0 = min_side
            resize_shape_1 = np.ceil((image.shape[1] * 1.0 / image.shape[0]) * min_side)
            return resize(image, (resize_shape_0, resize_shape_1))
#
# class MultiOutModel(SingleModel):
#     def __init__(self,model_def,weights,transformer,crop_num,min_side,crop_size):
#         super(MultiOutModel,self).__init__(model_def,weights,transformer,crop_num,min_side,crop_size)
#
#     def forward(self,images,key,repeat_num=1):
#         self.preprocess(images)
#         mini_batchsize = self.crop_num / repeat_num
#         score_means = []
#         total_score_means=[]
#         total_labels=[]
#         for n_rep in range(repeat_num):
#             self.net.blobs['data'].reshape(mini_batchsize, 3, self.crop_size, self.crop_size)
#             self.net.blobs['data'].data[...] = self.transformed_image[n_rep*mini_batchsize:(n_rep+1)*mini_batchsize]
#             self.net.forward()
#
#             scores = self.net.blobs['prob%d' % (key)].data
#             total_score_means.append(np.mean(self.net.blobs['fc100'].data,axis=0))
#             score_means.append(np.mean(scores, axis=0))
#         score = np.mean(np.asarray(score_means),axis=0)
#         total_score = np.mean(np.asarray(total_score_means),axis=0)
#         label_index = np.argmax(score) - 1  # we use 0 as negtive label
#         refine_label = int(subclass_dict[key][label_index])
#
#         total_label = np.argmax(total_score)
#
#         return refine_label,total_label
#

def load_validation():
    val_list_dir = os.path.join(Project_dir,"dataset/train")
    img_list = []
    label_list = []
    val_text = os.path.join(val_list_dir, 'val_list.txt')
    with open(val_text, 'r') as val_f:
        line = val_f.readline()
        while (line):
            line = line.strip()
            cell = line.split(' ')
            img_list.append(cell[0])
            label_list.append(cell[1])
            line = val_f.readline()
    val_f.close()
    return  img_list,label_list

def test_img(models,model_weights,image_paths,crop_image_paths=None,repeat_num=1):
    """
    :type image_paths: list
    """
    num_class = 97
    num_batch = len(image_paths)
    prob_averages = np.zeros((num_batch,num_class))
    images = []
    crop_images=[]
    if(crop_image_paths!=None):
        for crop_image_path in crop_image_paths:
            crop_image = caffe.io.load_image(crop_image_path)
            crop_images.append(crop_image)
    for image_path in image_paths:
        image = caffe.io.load_image(image_path)
        images.append(image)
    labels_=[]
    for index,model in enumerate(models):
        label,probs = model.forward(images,crop_images,'prob',repeat_num)
        prob_averages = prob_averages + np.asarray(probs) * model_weights[index]
        print label,
        labels_.append(label)
    vote_labels = np.argmax(prob_averages,axis=1)
    print ''

    return vote_labels

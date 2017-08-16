import os
import numpy as np
import sys
Project_dir = '/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS'

caffe_python = os.path.join(Project_dir,"caffe/build/install/python")
sys.path.insert(0,caffe_python)
import caffe
from skimage.transform import resize

test_dir=os.path.join(Project_dir,"dataset/train/subclass")

""" Load the net in the test phase for inference, and configure input preprocessing."""
model_def = os.path.join(Project_dir,"finetune/model/deploy/resnet50_deploy.prototxt")
model_weights=os.path.join(Project_dir,"finetune/model/finetune_models/cam_resnet50.caffemodel")


net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123])) # mean pixel
transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order


def resize_to_min_side(image,min_side=256):
	if(image.shape[0]>image.shape[1]):
		resize_shape_1 = min_side
		resize_shape_0 = np.ceil((image.shape[0]*1.0/image.shape[1])*min_side)
		return resize(image,(resize_shape_0,resize_shape_1))
	else:
		resize_shape_0 = min_side
		resize_shape_1 = np.ceil((image.shape[1]*1.0/image.shape[0])*min_side)
		return resize(image,(resize_shape_0,resize_shape_1))

def test_img(img_path):
	image_resize = 224
	net.blobs['data'].reshape(1,3,image_resize,image_resize)
	image = caffe.io.load_image(img_path)
	image_ = resize_to_min_side(image,256)
	center = np.array(image_.shape[:2]) / 2.0
	crop = np.tile(center, (1, 2))[0] + np.concatenate([-np.array([image_resize, image_resize]) / 2.0,np.array([image_resize, image_resize]) / 2.0])
	crop = crop.astype(int)
	input_ = image_[crop[0]:crop[2], crop[1]:crop[3], :]
	transformed_image = transformer.preprocess('data', input_)
	net.blobs['data'].data[...] = transformed_image

	# Forward pass.
	class_score =net.forward()['prob']
	label = np.argmax(class_score)
	return label,np.mean(class_score, axis=0)


class_dirs = os.listdir(test_dir)
class_dirs.sort()
top_n=10
for class_dir in class_dirs:
	test_files = [im for im in os.listdir(os.path.join(test_dir,class_dir))]
	test_files.sort()
	img_index = 0
	acc_cls=0;
	sum_class_score=np.zeros(100)
	analaysis_file = open(os.path.join(Project_dir,'utils/analysis/acc_reports.txt'),'a')
	for img in test_files:
		label,class_score= test_img(os.path.join(test_dir,class_dir,img))
		if(int(class_dir)==label):
			acc_cls = acc_cls + 1
		img_index = img_index + 1
		sum_class_score = sum_class_score+class_score
	sum_class_score = sum_class_score/len(test_files)
	ascending_order = np.argsort(sum_class_score)
	IDX_category = ascending_order[::-1]
	acc_cls = acc_cls*1.0/len(test_files)
	print "class %s acc:%f"%(class_dir,acc_cls),"Top 10: ",IDX_category[0:top_n],"with prob: ",sum_class_score[IDX_category[0:top_n]]
	print >>analaysis_file,"class %s acc:%f"%(class_dir,acc_cls),"Top 10: ",IDX_category[0:top_n],"with prob: ",sum_class_score[IDX_category[0:top_n]]
	analaysis_file.close()

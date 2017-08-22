import os
import numpy as np
import sys
import matplotlib.pyplot as plt
Project_dir = '/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS'

caffe_python = os.path.join(Project_dir,"caffe/build/install/python")
sys.path.insert(0,caffe_python)
import caffe
from skimage.transform import resize
from sklearn.metrics import log_loss
caffe.set_mode_gpu()
caffe.set_device(0)

test_dir=os.path.join(Project_dir,"dataset/train/subclass")

""" Load the net in the test phase for inference, and configure input preprocessing."""
model_def = os.path.join(Project_dir,"finetune/model/deploy/inception-v4_deploy.prototxt")
model_weights=os.path.join(Project_dir,"finetune/model/snapshot/inception-v4_centerloss/_iter_5200.caffemodel")

num_class=97
net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)
min_side=320
crop_size=299

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]

transformer = caffe.io.Transformer({'data': [1,3,299,299]})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([128,128,128]))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data',255)
transformer.set_input_scale('data',1.0/128.0)

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
	image_resize = crop_size
	net.blobs['data'].reshape(1,3,image_resize,image_resize)
	image = caffe.io.load_image(img_path)
	image_ = resize_to_min_side(image,min_side)
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
acc_ClassDistribution={}
loss_ClassDistribution={}
for class_dir in class_dirs:
	test_files = [im for im in os.listdir(os.path.join(test_dir,class_dir))]
	test_files.sort()
	img_index = 0
	acc_cls=0;
	sum_class_score=np.zeros(num_class)
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
	#print "class %s acc:%f"%(class_dir,acc_cls),"Top 10: ",IDX_category[0:top_n],"with prob: ",sum_class_score[IDX_category[0:top_n]]
	print >>analaysis_file,"class %s acc:%f"%(class_dir,acc_cls),"Top 10: ",IDX_category[0:top_n],"with prob: ",sum_class_score[IDX_category[0:top_n]]
	analaysis_file.close()
	acc_ClassDistribution[int(class_dir)] = acc_cls
	
	label = np.zeros(num_class)
	label[int(class_dir)]=1.0
	loss = log_loss(label,sum_class_score)
	loss_ClassDistribution[int(class_dir)] = loss
	print "class %s acc:%f loss:%f"%(class_dir,acc_cls,loss)
	

print acc_ClassDistribution
print loss_ClassDistribution
#Plot ACC
name = acc_ClassDistribution.keys()
acc = acc_ClassDistribution.values()

n_class = len(acc_ClassDistribution)
bar_width = 0.3
rect = plt.bar(name,acc,bar_width)
plt.xlabel('Index of Class')
plt.ylabel('trainACC ClassDistribution')
plt.show(block=True)

#Plot Loss
name = loss_ClassDistribution.keys()
loss = loss_ClassDistribution.values()

n_class = len(loss_ClassDistribution)
bar_width = 0.3
rect = plt.bar(name,loss,bar_width)
plt.xlabel('Index of Class')
plt.ylabel('trainloss ClassDistribution')
plt.show(block=True)

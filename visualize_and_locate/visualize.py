import numpy as np
import sys
import os

Project_dir="/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS"
caffe_root=os.path.join(Project_dir,"caffe/build/install")
sys.path.insert(0,os.path.join(Project_dir,"caffe/build/install/python"))

import caffe
from CAM.py_returnCAMmap import py_returnCAMmap
from CAM.py_map2jpg import py_map2jpg
import scipy.io
import matplotlib.pyplot as plt
from skimage.transform import resize

def im2double(im):
	im2 = (im - np.min(im))/(np.max(im)-np.min(im))
	return im2


#net_model = os.path.join(Project_dir,"finetune/model/deploy/resnet50_deploy.prototxt")
#net_weights=os.path.join(Project_dir,"finetune/model/finetune_models/cam_resnet50.caffemodel")
net_model = os.path.join(Project_dir,'finetune/model/deploy/inception-v4_deploy.prototxt')
net_weights= os.path.join(Project_dir,'finetune/model/snapshot/inception-v4_centerloss/_iter_5200.caffemodel')

out_layer = 'fc100'
crop_size=299
resize_shape=320
last_conv = 'inception_c3_concat'

# load CAM model and extract features
net = caffe.Net(net_model, net_weights, caffe.TEST)

#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2,0,1))
#transformer.set_mean('data', np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
#transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
#transformer.set_raw_scale('data',255)

transformer = caffe.io.Transformer({'data': [1,3,299,299]})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.array([128,128,128]))
transformer.set_channel_swap('data', (2,1,0))
transformer.set_raw_scale('data',255)
transformer.set_input_scale('data',1.0/128.0)

weights_LR = net.params[out_layer][0].data # get the softmax layer of the network

def show_CAM(image_path):
	image = caffe.io.load_image(image_path)
	image = resize(image, (resize_shape, resize_shape))

	# Take center crop.
	center = np.array(image.shape[:2]) / 2.0
	crop = np.tile(center, (1, 2))[0] + np.concatenate([
		-np.array([crop_size, crop_size]) / 2.0,
		np.array([crop_size, crop_size]) / 2.0
	])
	crop = crop.astype(int)
	input_ = image[crop[0]:crop[2], crop[1]:crop[3], :]

	# extract conv features
	net.blobs['data'].reshape(*np.asarray([1,3,crop_size,crop_size])) # run only one image
	net.blobs['data'].data[...][0,:,:,:] = transformer.preprocess('data', input_)
	out = net.forward()
	scores = out['prob']
	activation_lastconv = net.blobs[last_conv].data

	## Class Activation Mapping

	topNum = 5 # generate heatmap for top X prediction results
	scoresMean = np.mean(scores, axis=0)
	ascending_order = np.argsort(scoresMean)
	IDX_category = ascending_order[::-1] # [::-1] to sort in descending order

	curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:topNum],:])

	curResult = im2double(image)
	print "Top 5: ",IDX_category[0:topNum]
	plt.subplot(2,3,1)
	plt.imshow(image)
	plt.axis('off')

	for j in range(topNum):
		# for one image
		curCAMmap_crops = curCAMmapAll[:,:,j]
		curCAMmapLarge_crops = resize(curCAMmap_crops, (resize_shape,resize_shape))
		curHeatMap = resize(im2double(curCAMmapLarge_crops),(resize_shape,resize_shape)) # this line is not doing much
		curHeatMap = im2double(curHeatMap)

		curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
		curHeatMap = im2double(image)*0.2+im2double(curHeatMap)*0.7
	
		#plt.imshow(categories['categories'][IDX_category[j]][0][0], curHeatMap)
		plt.subplot(2,3,j+2)
		plt.imshow(curHeatMap)
		plt.axis('off')
		plt.text(0,0,"%d: %.1f"%(IDX_category[j],scoresMean[IDX_category[j]]))
	plt.show(block=True)
	plt.clf()

test_dir=os.path.join(Project_dir,"dataset/test-01/image")
imagelist = os.listdir(test_dir)
imagelist.sort()
for image in imagelist:
	show_CAM(os.path.join(test_dir,image))

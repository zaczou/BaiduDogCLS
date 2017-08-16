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
from CAM.py_generate_bbox import draw_bbox

caffe.set_device(5)

def im2double(im):
	im2 = (im - np.min(im))/(np.max(im)-np.min(im))
	return im2


net_model = os.path.join(Project_dir,"finetune/model/deploy/resnet50_deploy.prototxt")
net_weights=os.path.join(Project_dir,"finetune/model/finetune_models/cam_resnet50.caffemodel")

out_layer = 'fc100'
crop_size=224
last_conv = 'res5c'

# load CAM model and extract features
net = caffe.Net(net_model, net_weights, caffe.TEST)

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', np.load(caffe_root + '/python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))
transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB
transformer.set_raw_scale('data',255)

weights_LR = net.params[out_layer][0].data # get the softmax layer of the network

def test_img(image_path,show_cam=False):
	image = caffe.io.load_image(image_path)
	image_raw_shape=image.shape
	image = resize(image, (256, 256))

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

	if(show_cam==False):
		return IDX_category[0]

	curCAMmapAll = py_returnCAMmap(activation_lastconv, weights_LR[IDX_category[:topNum],:])

	curResult = im2double(image)
	print "Top 5: ",IDX_category[0:topNum]
	plt.figure(1)
	plt.clf()
	plt.subplot(2,3,1)
	plt.imshow(image)
	plt.axis('off')

	for j in range(topNum):
		# for one image
		curCAMmap_crops = curCAMmapAll[:,:,j]
		curCAMmapLarge_crops = resize(curCAMmap_crops, (256,256))
		curHeatMap = resize(im2double(curCAMmapLarge_crops),(256,256)) # this line is not doing much
		curHeatMap = im2double(curHeatMap)
		map_resize = resize(curHeatMap,image_raw_shape[0:2])
		if(j==0):
			plt.imsave('heatmap.jpg',map_resize)

		curHeatMap = py_map2jpg(curHeatMap, None, 'jet')
		curHeatMap = im2double(image)*0.2+im2double(curHeatMap)*0.7
	
		#plt.imshow(categories['categories'][IDX_category[j]][0][0], curHeatMap)
		plt.subplot(2,3,j+2)
		plt.imshow(curHeatMap)
		plt.axis('off')
		plt.text(0,0,"%d: %.1f"%(IDX_category[j],scoresMean[IDX_category[j]]))

	plt.show(block=False)
	return IDX_category[0],map_resize,


#=============
#Crop train_img
#=============
train_src=os.path.join(Project_dir,"dataset/train/IMG")
train_crop=os.path.join(Project_dir,"dataset/train/crop_image")
train_files = [im for im in os.listdir(train_src)]
train_files.sort()
img_index = 0

if(not os.path.isdir(train_crop)):
	os.mkdir(train_crop)

for img in train_files:
	image_path = os.path.join(train_src,img)
	label,map_resize = test_img(image_path,True)

	# crop_img = draw_bbox(image_path,'heatmap.jpg',2)
	# plt.imsave(os.path.join(head_crop,img),crop_img)

	crop_img = draw_bbox(image_path, 'heatmap.jpg', 1)
	plt.imsave(os.path.join(train_crop, img), crop_img)
	print "save crop_image to",os.path.join(train_crop, img)

	#plt.figure(2)
	#plt.imshow(crop_img)
	#plt.show(block=True)
	#print label


#=============
#Crop test_img
#=============
test_src=os.path.join(Project_dir,"dataset/test/image")
test_crop=os.path.join(Project_dir,"dataset/test/crop_image")
test_files = [im for im in os.listdir(test_src)]
test_files.sort()
img_index = 0
for img in test_files:
	image_path = os.path.join(test_src,img)
	label,map_resize = test_img(image_path,True)

	# crop_img = draw_bbox(image_path,'heatmap.jpg',2)
	# plt.imsave(os.path.join(head_crop,img),crop_img)

	crop_img = draw_bbox(image_path, 'heatmap.jpg', 1)
	plt.imsave(os.path.join(test_crop, img), crop_img)
	print "save crop_image to",os.path.join(test_crop, img)

	#plt.figure(2)
	#plt.imshow(crop_img)
	#plt.show(block=True)
	#print label

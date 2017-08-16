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
def area(rect):
	return (rect[2]-rect[0])*(rect[3]-rect[1])

"""
index_crop
  0: most image 
  1:doge detect 
  2:head detect
"""
def draw_bbox(curImgFile,curHeatMapFile,index_crop):
	bbox_threshold = [20, 100, 110] # parameters for the bbox generator#[20, 100, 110]
	curParaThreshold = str(bbox_threshold[0])+' '+str(bbox_threshold[1])+' '+str(bbox_threshold[2])+' '
	#curHeatMapFile = 'bboxgenerator/heatmap_6.jpg';
	#curImgFile = 'bboxgenerator/sample_6.jpg';
	curBBoxFile = 'CAM/bboxgenerator/heatmap_rect.txt';

	os.system("CAM/bboxgenerator/./dt_box "+curHeatMapFile+' '+curParaThreshold+' '+curBBoxFile)

	with open(curBBoxFile) as f:
		for line in f:
			items = [int(x) for x in line.strip().split()]

	boxData1 = np.array(items[0::4]).T
	boxData2 = np.array(items[1::4]).T
	boxData3 = np.array(items[2::4]).T
	boxData4 = np.array(items[3::4]).T

	boxData_formulate = np.array([boxData1, boxData2, boxData1+boxData3, boxData2+boxData4]).T

	col1 = np.min(np.array([boxData_formulate[:,0], boxData_formulate[:,2]]), axis=0)
	col2 = np.min(np.array([boxData_formulate[:,1], boxData_formulate[:,3]]), axis=0)
	col3 = np.max(np.array([boxData_formulate[:,0], boxData_formulate[:,2]]), axis=0)
	col4 = np.max(np.array([boxData_formulate[:,1], boxData_formulate[:,3]]), axis=0)

	boxData_formulate = np.array([col1, col2, col3, col4]).T

	curHeatMap =  caffe.io.load_image(curHeatMapFile,0)
	curImg =  caffe.io.load_image(curImgFile)

	curHeatMap=np.reshape(curHeatMap,(curHeatMap.shape[0],curHeatMap.shape[1]))
	curHeatMap = im2double(curHeatMap)
	curHeatMap = py_map2jpg(curHeatMap, None, 'jet')

	curImg = resize(curImg,curHeatMap.shape)
	curHeatMap = im2double(curImg)*0.2+im2double(curHeatMap)*0.7
	colors = plt.cm.hsv(np.linspace(0,1,21)).tolist()


	#for i in range(boxData_formulate.shape[0]): # for each bbox
	#	print(boxData_formulate[i][:2])
	#	print(boxData_formulate[i][2:])
	#	area_per = area(boxData_formulate[i])*1.0/(curImg.shape[0]*curImg.shape[1])
	#	"""if(area_per<0.1 or area_per>0.6):
	#		continue
	#	print area_per
	#	plt.imshow(curHeatMap)
	#	currentAxis = plt.gca()
	#    	coords=(boxData_formulate[i][0],boxData_formulate[i][1]),boxData_formulate[i][2]-boxData_formulate[i][0]+1,boxData_formulate[i][3]-boxData_formulate[i][1]+1
	#    	currentAxis.add_patch(plt.Rectangle(*coords,fill=False,edgecolor=colors[0],linewidth=2))
	#	plt.show(block=True)

	#	return curImg[boxData_formulate[i][1]:boxData_formulate[i][3],boxData_formulate[i][0]:boxData_formulate[i][2]]
	#	"""
	return curImg[boxData_formulate[index_crop][1]:boxData_formulate[index_crop][3],boxData_formulate[index_crop][0]:boxData_formulate[index_crop][2]]

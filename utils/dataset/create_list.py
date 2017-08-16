import os
import numpy as np
import shutil
project_dir = "/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS"

dataset_dir = os.path.join(project_dir,"dataset/train")

train_text =  os.path.join(dataset_dir,"data_train_image.txt")
val_text = os.path.join(dataset_dir,"val.txt")

# project 133 classes to 100 classes
def decode(encode_dict,label):
	return encode_dict[label]
def encode(encode_dict,label):
	return encode_dict.keys()[encode_dict.values().index(label)]

def save_mapdict(encode_dict):
	encode_dict_file=open(os.path.join(dataset_dir,"map_dict"),"w")
	for key,value in encode_dict.items():
		print >>encode_dict_file,"%s %s"%(key,value)
def load_mapdict():
	encode_dict_file=open(os.path.join(dataset_dir,"map_dict"),"r")
	encode_dict={}
	line = encode_dict_file.readline()
	while (line):
		line = line.strip()
		cell = line.split(' ')
		encode_dict[cell[0]]=cell[1]
		line = encode_dict_file.readline()

	encode_dict_file.close() 
	return encode_dict


#load imagelist and move image to different dir/each dir for a single class
def load_imagelist():
	img_list=[]
	label_list=[]
	raw_dict_list=[]
	#load traing text_file and val
	with open(train_text,'r') as train_f:
		line = train_f.readline()
		while (line):
			line = line.strip()
			cell = line.split(' ')
			img_list.append(cell[0])
			label_list.append(cell[1])
			if(cell[1] not in raw_dict_list):
				raw_dict_list.append(cell[1])
			line = train_f.readline()    
		train_f.close() 

	with open(val_text,'r') as val_f:
		line = val_f.readline()
		while (line):
			line = line.strip()
			cell = line.split(' ')
			img_list.append(cell[0])
			label_list.append(cell[1])
			if(cell[1] not in raw_dict_list):
				raw_dict_list.append(cell[1])
			line = val_f.readline()    
		val_f.close() 
	save_mapdict(dict(enumerate(raw_dict_list)))
	map_dict=load_mapdict()

	return img_list,label_list,map_dict


#move data to different dir/each dir for a single class
def movefile(img_list,label_list,map_dict):
	src_dir = os.path.join(dataset_dir,"IMG")
	if(not os.path.isdir(src_dir)):
		os.mkdir(os.path.join(dataset_dir,"subclass"))
	for index,image_file in enumerate(img_list):
		dst_dir = os.path.join(dataset_dir,"subclass",encode(map_dict,label_list[index]))
		if(not os.path.isdir(dst_dir)):
			os.mkdir(dst_dir)
		src_path =os.path.join(src_dir,image_file+'.jpg')
		if os.path.exists(src_path):
			shutil.copy(src_path,dst_dir)
		else:
			print image_file,".jpg is not found"

#generate datalist
img_list,label_list,map_dict = load_imagelist()

np.random.seed(100)
random_index = np.arange(len(img_list))
np.random.shuffle(random_index)
train_num=0.8*len(img_list)

train_list_name = os.path.join(dataset_dir,"train_list.txt")
val_list_name = os.path.join(dataset_dir,"val_list.txt")
train_list_file=open(train_list_name,"w")
val_list_file=open(val_list_name,"w")

for i in range(len(img_list)):
	if(i<train_num):
		img_path = img_list[random_index[i]]+'.jpg'
		print >>train_list_file,"%s %s"%(img_path,encode(map_dict,label_list[random_index[i]]))
	else:
		img_path = img_list[random_index[i]]+'.jpg'
		print >>val_list_file,"%s %s"%(img_path,encode(map_dict,label_list[random_index[i]]))		
train_list_file.close()
val_list_file.close()

movefile(img_list,label_list,map_dict)

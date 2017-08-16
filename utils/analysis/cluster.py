
def is_in_dict(classes,value):
	if len(classes)==0:
		return 0,-1
	for key,value_list in enumerate(classes):
		if((value in classes[key])):
			return 1,key
	return 0,-1

top_num=10
def cluster(threshold):
	acc=open("acc_analaysis.txt","r")
	line = acc.readline()
	classes=dict()
	num_class=0
	while (line):
		line = line.strip()
		cell = line.split(' ')
		if(float(cell[2][4:-1])<=0.85):#0.85
			print cell[1],": ",cell[2][4:-1]
			# is_in_list,key=is_in_dict(classes,cell[1])
			# if(not is_in_list):
			# 	classes[num_class]=[cell[1]]
			# 	key = num_class
			# 	num_class = num_class + 1
			cur_list=[]
			cur_key=0;
			# for index,value in enumerate(cell[7:11]):
			# 	if(len(classes[key])>0 and (value not in classes[key])):
			# 		if(float(cell[16+index])>threshold):
			# 			classes[key].append(value)
			for index,value in enumerate(cell[6:6+top_num]):
				if(float(cell[10+top_num+index])>threshold):
					cur_list.append(value)
			is_in_list=0
			for v in cur_list:
				is_in_list,key=is_in_dict(classes,v)
				if(is_in_list):
					cur_key = key
					break;
			if(not is_in_list):
			 	classes[num_class]=[cell[1]]
			 	key = num_class
			 	cur_key = key
			 	num_class = num_class + 1
			for v in cur_list:
				is_in,key=is_in_dict(classes,v)
				if(not is_in):
					classes[cur_key].append(v)


		line = acc.readline()
	acc.close() 
	total_num=0
	for key,value_list in enumerate(classes):
		print key,classes[key]
		total_num = total_num + len(classes[key])
	print total_num
cluster(0.01)#0.2

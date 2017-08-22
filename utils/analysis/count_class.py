import os
import numpy as np
Project_dir = "/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS"
import matplotlib.pyplot as plt
def load_train_list(train_list_file):
	with open(train_list_file,"r") as f:
		lines = f.readline()
		ClassDistribution={}
		while(lines):
			temp = lines.strip()
			lineCell = temp.split(' ')
			if(lineCell[1] not in ClassDistribution):
				key = lineCell[1]
				ClassDistribution[key] = 1
			else:
				ClassDistribution[lineCell[1]] = ClassDistribution[lineCell[1]] + 1
			lines = f.readline()
	return ClassDistribution

ClassDistribution = load_train_list(os.path.join(Project_dir,"dataset/train/train_list.txt"))

print ClassDistribution
#Plot 
name = ClassDistribution.keys()
nums = ClassDistribution.values()

n_class = len(ClassDistribution)
bar_width = 0.3
rect = plt.bar(name,nums,bar_width)
plt.xlabel('Index of Class')
plt.ylabel('ClassDistribution of training set')
plt.show()
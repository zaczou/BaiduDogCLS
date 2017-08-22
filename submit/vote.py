# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:38:12 2017

@author: xiong
"""
import time

file1 = "submit-0822.txt" #0.1953
file2 = "submit-0820-pseudo-ensemble3.txt" #0.197
file3 = "submit-0819-pseudo-ensemble.txt" #0.1974
file4= "submit-0822-single.txt" #0.1988

f1 = open(file1,'r')
f2 =  open(file2,'r')
f3 = open(file3,'r')
f4 = open(file4,'r')

files=[f1,f2,f3,f4]

f_fin = open("last_commit.txt","w")#0.1952
import numpy as np
count=0
for i in range(29282):
    labels=[]
    for f in files:
        line = f.readline().strip().split('\t')
        fileid = line[1]
        #print line[0],
        labels.append(line[0])
    #print "" 
    label_candidate= set(labels)
    label_count={}
    for key in label_candidate:
        label_count[key] = 0
        for label in labels:
            if(key==label):
                label_count[key] = label_count[key] + 1 
    index = np.argmax(label_count.values())
	
	#（各占一半时,最相信目前的最高分的label)
    if(label_count[labels[0]]==2):
        vote = labels[0]
    else:
        vote = label_count.keys()[index]
    #vote label
    if(len(set(labels))>1):
        print labels,vote
        count =count +1
    print>>f_fin,"%s\t%s"%(vote,fileid)
    
print "total count",count
for f in files:
    f.close()
f_fin.close()
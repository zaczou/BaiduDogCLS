import numpy as np
import sys
import os
import time
Project_dir="/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS"

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


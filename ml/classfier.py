import lmdb
import numpy as np
import sys
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
Project_dir="/home/lab-xiong.jiangfeng/Projects/BaiduDogCLS"
caffe_root=os.path.join(Project_dir,"caffe/build/install")
sys.path.insert(0,os.path.join(Project_dir,"caffe/build/install/python"))
import caffe
from utils import *
import pandas as pd
from sklearn.model_selection import KFold,StratifiedKFold

import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional     scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search

datum = caffe.proto.caffe_pb2.Datum()



#Convert LMDB to npy format
def load_dataset(lmdb_file):
    lmdb_env = lmdb.open(lmdb_file)
    lmdb_txn = lmdb_env.begin()
    lmdb_cursor = lmdb_txn.cursor()
    dataset=[]
    for key,value in lmdb_cursor:
        datum.ParseFromString(value)
        data = caffe.io.datum_to_array(datum)
        dataset.append(data)
        print "Loading ",key
    return np.asarray(dataset)

#save to numpy dataset format to speed up loading
import os
def convert_lmdb_to_npy(prefix):
    train_lmdbdir=os.path.join(Project_dir,"dataset/train/feat")
    dataset_x= load_dataset(os.path.join(train_lmdbdir,"%s_train"%(prefix)))[:,:,0,0]
    dataset_y= load_dataset(os.path.join(train_lmdbdir,"%s_label"%(prefix)))[:,0,0,0]
    #
    val_dataset_x= load_dataset(os.path.join(train_lmdbdir,"%s_val"%(prefix)))[:,:,0,0]
    val_dataset_y= load_dataset(os.path.join(train_lmdbdir,"%s_label_val"%(prefix)))[:,0,0,0]

    np.save("features/dataset_%s"%(prefix),dataset_x)
    np.save("features/dataset_%s_label"%(prefix),dataset_y)
    np.save("features/dataset_%s_val"%(prefix),val_dataset_x)
    np.save("features/dataset_%s_label_val"%(prefix),val_dataset_y)

    dataset_test01 = load_dataset(os.path.join(Project_dir,'dataset/test-01/feat/%s'%(prefix)))
    np.save('features/test_%s'%(prefix), dataset_test01)

def load_npy(prefix):
    dataset_x_train = np.load("features/dataset_%s.npy"%(prefix))
    dataset_y_train = np.load("features/dataset_%s_label.npy"%(prefix))
    dataset_x_val = np.load("features/dataset_%s_val.npy" % (prefix))
    try:
        dataset_y_val = np.load("features/dataset_%s_label_val.npy"%(prefix))
    except:
        dataset_y_val=0
        return dataset_x_train, dataset_y_train, dataset_x_val, dataset_y_val
    return dataset_x_train, dataset_y_train,dataset_x_val,dataset_y_val

def load_test(prefix):
    test_01 = np.load('features/test_%s.npy'%(prefix))
    return test_01

# convert_lmdb_to_npy('resnet50')
# convert_lmdb_to_npy('resnext224')
# convert_lmdb_to_npy('resnext299')
# convert_lmdb_to_npy('inception-v4')

#test_01_x1,test_01_x2,test_01_x3,test_01_x4 =load_test('resnet50'),load_test('resnext224'),load_test('resnext299'),load_test('inception-v4')

train_x1,train_y1,val_x1,val_y1=load_npy('resnet50')
train_x2,train_y2,val_x2,val_y2=load_npy('resnext224')
train_x3,train_y3,val_x3,val_y3=load_npy('resnext299')
train_x4,train_y4,val_x4,val_y4=load_npy('inception-v4')

# X_train,y_train = np.concatenate((train_x1,train_x2,train_x3,train_x4),axis=1),train_y1
#
# X_val,y_val = np.concatenate((val_x1,val_x2,val_x3,val_x4),axis=1),val_y1

X_train,y_train = train_x4,train_y1
X_val,y_val = val_x4,val_y1


#Linear regression with L2Norm #default 0.835---online 0.216
#0.835349(Resnet50) #0.853578(ResNext299) #combineed(0.8609)---online0.1934

# test_01=np.append(inception_test_01,resnext299_test_01,axis=1)
# dataset_x=np.append(inception_dataset_x,inception_crop_dataset_x,axis=1)
# dataset_y = inception_dataset_y
#
# #dataset_x=inception_dataset_x
# #dataset_y = inception_dataset_y
# test_01 = inception_test_01
# X_train, X_test, y_train, y_test = train_test_split(dataset_x, dataset_y, test_size=0.2, random_state=100)
#
ridge_cls = RidgeClassifier(alpha=1e-5)
print X_train.shape,y_train.shape
ridge_cls.fit(X_train,y_train)
ridge_cls.predict(X_val)
acc=ridge_cls.score(X_val,y_val)
print "using LR on validation acc:%.3f error:%.3f"%(acc,1-acc) #on validation 0.9108

def forward(net,input):
    net.blobs['resnet50'].data[...] = input[0]
    net.blobs['resnext224'].data[...] = input[1]
    net.blobs['resnext299'].data[...] = input[2]
    net.blobs['inception-v4'].data[...] = input[3]
    net.forward()
    score = net.blobs['prob'].data
    label = np.argmax(score)
    return label

def submit():
    encode_dict = load_mapdict()
    test_dir_1 = os.path.join(Project_dir,'dataset/test-01/test-01_list.txt')
    submit_file = open(os.path.join(Project_dir,'submit/submit-0813.txt', 'w'))
    f=open(test_dir_1,'r')
    for index,line in enumerate(f):
        line_split = line.strip()
        line_cell = line_split.split(' ')
        file_name = line_cell[0].split('/')[-1][:-4]
        print file_name,
        prdict_label=forward(net,[test_01_x1[index],test_01_x2[index],test_01_x3[index],test_01_x4[index]])
        decode_label = decode(encode_dict,str(int(prdict_label)))
        print " predict: ",prdict_label," decode: ",decode_label
        print >> submit_file, "%s\t%s" % (decode_label, file_name)
#submit()

def runXGB(train_X, train_y, test_X, test_y=None, feature_names=None, seed_val=0, numround=1000):
    params = {}
    params['eval_metric'] = 'merror'
    params['objective'] = 'multi:softmax'
    params['num_class'] = 100
    params["eta"] = 0.003
    #params["eta"] = 0.01
    params["subsample"] = 0.9
    params["min_child_weight"] = 5
    params["colsample_bytree"] = 0.4
    params["max_depth"] = 8
    params["silent"] = 1
    params['booster'] = 'gblinear'
    params['xgb_model'] = 'xgboost'
    params['save_period'] = 5
    params['model_dir'] = './snapshot'
    params['lambda'] = 1
    params['alpha'] = 5
    params['rate_drop'] = 0.2
    params["seed"] = seed_val
    num_rounds = numround
    plst = list(params.items())
    xgtrain = xgb.DMatrix(train_X, label=train_y)
    print params
    if test_y is not None:
        xgtest = xgb.DMatrix(test_X, label=test_y)
        watchlist = [(xgtrain, 'train'), (xgtest, 'test')]
        model = xgb.train(plst, xgtrain, num_rounds, watchlist, early_stopping_rounds=100, verbose_eval=5)
    else:
        xgtest = xgb.DMatrix(test_X)
        model = xgb.train(plst, xgtrain, num_rounds)


    return model
#runXGB(X_train,y_train,X_val,y_val)
#test and submit
#submit()


#K-means for cluster
# from sklearn.cluster import KMeans
# from sklearn.metrics import accuracy_score
# cluster_centers = np.zeros((100,1536))
# for i in range(100):
#     cluster_centers[i] = np.mean(X_train[y_train==i],axis=0)
# kmeans  = KMeans(n_clusters=100,init=cluster_centers,random_state=0,verbose=1,max_iter=1000,n_jobs=-1,tol=1e-8).fit(X_train)
# label = kmeans.predict(X_val)
# acc= accuracy_score(y_val,label)
# print acc
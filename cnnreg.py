#coding: utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import time
import pylab
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import os
from PIL import Image
import random
import scipy.stats
import cPickle
import csv
import cv2

gpu_flag = 0

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

batchsize = 6
val_batchsize = 4
n_epoch = 100
height=165
width=1200

# plt.imshow(X_train[1][0], cmap=pylab.cm.gray_r, interpolation='nearest')
# plt.show()
#, stride=1,pad=2
model = chainer.FunctionSet(conv1=L.Convolution2D(1,  10, ksize=(2,10),stride=1),
                            conv2=L.Convolution2D(10, 5,  ksize=(2,10),stride=1),
                            fc6=L.Linear(6000, 1024),
                            fc7=L.Linear(1024, 350),
                            fc8=L.Linear(350, 1),
                            )

if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv1(x))), ksize=(2,10),stride=(2,5))
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv2(h))), ksize=(2,10),stride=(2,5))
    h = F.dropout(F.relu(model.fc6(h)))
    h = F.dropout(F.relu(model.fc7(h)))
    h = model.fc8(h)
    if train:
        return F.mean_squared_error(h,t), h

    else:
        return F.mean_squared_error(h, t), h

optimizer = optimizers.RMSpropGraves()
optimizer.setup(model)

fp1 = open("accuracy.txt", "w")
fp2 = open("loss.txt", "w")
trainpic = open("train_list.txt", "w")
testpic = open("test_list.txt", "w")
gosalist = open("gosa.txt", "w")



fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")

train_data=[]
test_data=[]
#print "gyokaku.csv open"
gyokaku = np.genfromtxt("gyokaku.csv", delimiter=",", dtype=np.string_)
#all = np.random.permutation(gyokaku)
#all=gyokaku
#a=len(all)
#ds = np.arange(a)
#N_data, N_test_data = np.split(all, [int(a * 0.8)])

#np.save("N_data",N_data)
#np.save("N_test_data",N_test_data)


N_data = np.load("N_data.npy")
N_test_data = np.load("N_test_data.npy")

#print N_data,N_test_data
for al in N_data:
    data=np.load(str(al[0])+".npy").astype(np.float32) / 255
    train=[]
    for i in range(0,len(data)-width,width):
        flag=0
        for k in range(width):
            if np.count_nonzero(data[i+k,0:height])==0:
                flag+=1
        if flag<=100:
            resize = data[i:i+width, 0:height].reshape(1,width,height)
            train.append(resize)
    train_data.append((train,al[2]))   
        
for al in N_test_data:
    data=np.load(str(al[0])+".npy").astype(np.float32) / 255
    test=[]
    for i in range(0,len(data)-width,width):
        flag=0
        for k in range(width):
            if np.count_nonzero(data[i+k,0:height])==0:
                flag+=1
        if flag<=100:
            resize = data[i:i+width, 0:height].reshape(1,width,height)
            test.append(resize)
    test_data.append((test,al[2]))   
# 訓練ループ
start_time = time.clock()


#np.random.shuffle(train_data)
#np.random.shuffle(test_data)
N=0
N_test=0
print N,N_test

for epoch in range(1, n_epoch + 1):


    print "epoch: %d" % epoch

    sum_loss = 0
    train_gosa=0
    test_gosa=0
    count=0
    for i in train_data:
        batchsize=len(i[0])
        x_batch1 = np.ndarray(
            (batchsize, 1, width, height), dtype=np.float32)
        y_batch1 = np.ndarray((batchsize,), dtype=np.float32)
        batch_pool = [None] * batchsize
        path, label = i
        for z in range(batchsize):
            batch_pool[z] = path[z]
            x_batch1[z]=batch_pool[z]
            y_batch1[z] = label
            count+=1
        #x_batch2 = x_batch1.reshape(batchsize, 1, insize, insize)
        x_batch = xp.asarray(x_batch1)
        y_batch = xp.asarray(y_batch1).reshape(batchsize,1)
        optimizer.zero_grads()
        gosa, ans = forward(x_batch, y_batch)
        gosa.backward()
        optimizer.update()
        for sa in range(batchsize):
            answer = float(ans.data[sa]) - float(y_batch1[sa])
            #print answer,float(gyokaku[0][15])
            train_gosa += abs(answer*float(gyokaku[0][15]))
        #train_gosa += float(gosa.data) * len(y_batch)
    print "train mean squared error : %f" % (train_gosa / count)
    fp2.write("%d\t%f\n" % (epoch, train_gosa / count))
    fp2.flush()
    
    count = 0
    sum_accuracy = 0
    test_gosa = 0
    for i in test_data:
        val_batchsize = len(i[0])
        val_x_batch = np.ndarray(
            (val_batchsize, 1, width, height), dtype=np.float32)
        val_y_batch = np.ndarray((val_batchsize,), dtype=np.float32)
        val_batch_pool = [None] * val_batchsize

        path, label = i
        for zz in range(val_batchsize):
            val_batch_pool[zz] = path[zz]
            val_x_batch[zz]=val_batch_pool[zz]
            val_y_batch[zz] = label
            count+=1

        x_batch = xp.asarray(val_x_batch)
        y_batch = xp.asarray(val_y_batch).reshape(val_batchsize,1)

        gosa , ans = forward(x_batch, y_batch, train=False)
        #sum_accuracy += float(gosa.data) * len(val_y_batch)
        for sa in range(val_batchsize):
            answer = float(ans.data[sa])- float(val_y_batch[sa])
            test_gosa += abs(answer*float(gyokaku[0][15]))
        #pearson = scipy.stats.pearsonr(acc.data,ans.data)


    print "test gosa: %f" % (test_gosa / count)
    fp1.write("%d\t%f\n" % (epoch, test_gosa / count))
    fp1.flush()

end_time = time.clock()
print end_time - start_time

fp1.close()
fp2.close()
gosalist.close()

import cPickle
# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
model.to_cpu()
cPickle.dump(model, open("model.pkl", "wb"), -1)

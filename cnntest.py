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
val_batchsize=1
tate=165
yoko=1200


if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np


if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()
with open("model.pkl","rb") as i
    model = cPickle.load(i)
 

def forward(x_data, y_data, train=False):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv1(x))), 3,stride=2)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv2(h))), 3,stride=2)
    h = F.dropout(F.relu(model.fc6(h)))
    h = F.dropout(F.relu(model.fc7(h)))
    h = model.fc8(h)
    #import pdb; pdb.set_trace()
    return F.accuracy(h, t), h

optimizer = optimizers.RMSpropGraves()
optimizer.setup(model)

fp1 = open("accuracy.txt", "w")
fp2 = open("loss.txt", "w")


fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")

all_data=[]
train_data=[]
train_list=[]

def csvread(path):
for al in path:
    print path
    data = np.genfromtxt(str(al)+".csv", delimiter=",", dtype=np.int32)

    for i in range(1,4801,yoko):
        flag=0
        for k in range(yoko):
            if int(data[len(data)-i-k][0])==0:
                flag=1
                break
        if flag==0:
            test_list = np.ndarray((1, yoko, 165), dtype=np.uint8)
            test_list[0]=data[len(data)-yoko-i:len(data)-i, 0:165]
            size = (255, 255)
            resize = cv2.resize(test_list[0], size, interpolation=cv2.INTER_CUBIC)
            ans=0
            mo=float(al[2])
            if mo>100:
                ans=1
            if mo>201:
                ans=2
            if mo>401:
                ans=3
            if mo>700:
                ans=4
            count[ans]=count[ans]+1     
            all_data.append((resize,ans))


# 訓練ループ
start_time = time.clock()

a=len(all_data)
ds = np.arange(a)
N, N_test = np.split(ds, [int(ds.size * 0.8)])
N=len(N)

N_test=len(N_test)-1
print a,N,N_test

perm = np.random.permutation(len(all_data))
for epoch in range(1, n_epoch + 1):


    print "epoch: %d" % epoch

    sum_loss = 0
    train_gosa=0
    test_gosa=0
    count=0
    for i in range(0, N, batchsize):
        x_batch1 = np.ndarray(
            (batchsize, 1, 255, 255), dtype=np.float32)
        y_batch1 = np.ndarray((batchsize,), dtype=np.int32)
        batch_pool = [None] * batchsize

        for z in range(batchsize):
            path, label = all_data[perm[count]]
            batch_pool[z] = path
            x_batch1[z]=batch_pool[z]
            y_batch1[z] = label
            count += 1
        #x_batch2 = x_batch1.reshape(batchsize, 1, insize, insize)
        x_batch = xp.asarray(x_batch1)
        y_batch = xp.asarray(y_batch1)
        optimizer.zero_grads()
        gosa = forward(x_batch, y_batch)
        gosa.backward()
        optimizer.update()
        train_gosa += float(gosa.data) * len(y_batch)
    print "train loss : %f" % (train_gosa / N)
    fp2.write("%d\t%f\n" % (epoch, train_gosa / N))
    fp2.flush()

    sum_accuracy = 0
    test_gosa1=0
    for i in range(0, N_test, val_batchsize):
        val_x_batch = np.ndarray(
            (val_batchsize, 1, 255, 255), dtype=np.float32)
        val_y_batch = np.ndarray((val_batchsize,), dtype=np.int32)
        val_batch_pool = [None] * val_batchsize
        for zz in range(val_batchsize):
            path, label = all_data[perm[count]]
            val_batch_pool[zz] = path
            val_x_batch[zz]=val_batch_pool[zz]
            val_y_batch[zz] = label
            count+=1
        x_batch = xp.asarray(val_x_batch)
        y_batch = xp.asarray(val_y_batch)

        gosa , ans = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(gosa.data) * len(val_y_batch)
        #for sa in range(val_batchsize):
        #    test_gosa1+=abs(float(ans.data[sa]-val_y_batch[sa]))
        #pearson = scipy.stats.pearsonr(acc.data,ans.data)


    count=0
    print "test accuracy: %f" % (sum_accuracy / N_test)
    fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
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

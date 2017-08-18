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

gpu_flag = -1

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np

batchsize = 5
val_batchsize=17
n_epoch = 100
tate=165
yoko=25


# 画像を (nsample, channel, height, width) の4次元テンソルに変換
# MNISTはチャンネル数が1なのでreshapeだけでOK

# plt.imshow(X_train[1][0], cmap=pylab.cm.gray_r, interpolation='nearest')
# plt.show()
#, stride=1,pad=2
model = chainer.FunctionSet(conv1=L.Convolution2D(1,  40, 2),
                            conv2=L.Convolution2D(40, 20,  2),
                            fc6=L.Linear(4000, 2048),
                            fc7=L.Linear(2048, 512),
                            fc8=L.Linear(512, 1),
                            )
if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()

def forward(x_data, y_data, train=True):
    x, t = chainer.Variable(x_data), chainer.Variable(y_data)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv1(x))), 2,stride=2)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv2(h))), 3,stride=2)
    h = F.dropout(F.relu(model.fc6(h)))
    h = F.dropout(F.relu(model.fc7(h)))
    h = F.relu(model.fc8(h))
    #import pdb; pdb.set_trace()
    if train:
        return F.mean_squared_error(h, t)

    else:
        return F.mean_squared_error(h, t), h

optimizer = optimizers.RMSpropGraves()
optimizer.setup(model)

fp1 = open("accuracy.txt", "w")
fp2 = open("loss.txt", "w")
trainpic = open("train_list.txt", "w")
testpic = open("test_list.txt", "w")
gosa = open("gosa.txt", "w")



fp1.write("epoch\ttest_accuracy\n")
fp2.write("epoch\ttrain_loss\n")

all_data=[]
ans_data=[]

train_data=[]
val_data=[]
gyosyu_list=[]



train_list=[]

#print "gyokaku.csv open"
gyokaku = np.genfromtxt("gyokaku.csv", delimiter=",", dtype=np.string_)


for al1,al in enumerate(gyokaku):
    #print str(al[1])+".pkl open"
    with open(str(al[1]) + '.pkl', 'rb') as i:
        data = cPickle.load(i)

    for i in range(2,50):
        if int(data[len(data)-25*i][0])!=0:
            all_data.append((data[len(data)-25*(i+1):len(data)-25*i,0:],float(al[2])))




# 訓練ループ
start_time = time.clock()

a=len(all_data)
N=int(a*0.8)
N_test=int(a-N)
print a,N,N_test

for epoch in range(1, n_epoch + 1):


    print "epoch: %d" % epoch

    sum_loss = 0
    count=0
    for i in range(0, N, batchsize):
        if i%100==0:
            print i
        x_batch1 = np.ndarray(
            (batchsize, 1, yoko, tate), dtype=np.float32)
        y_batch1 = np.ndarray((batchsize,), dtype=np.float32)
        batch_pool = [None] * batchsize

        for z in range(batchsize):
            path, label = all_data[count]
            batch_pool[z] = path
            x_batch1[z]=batch_pool[z]
            y_batch1[z] = label
            count += 1
        #x_batch2 = x_batch1.reshape(batchsize, 1, insize, insize)
        x_batch = xp.asarray(x_batch1)
        y_batch = xp.asarray(y_batch1).reshape(batchsize,1)
        optimizer.zero_grads()
        loss = forward(x_batch, y_batch)
        loss.backward()
        optimizer.update()
        sum_loss += float(loss.data) * len(y_batch)
    print "train mean squared error : %f" % (sum_loss / N)
    fp2.write("%d\t%f\n" % (epoch, sum_loss / N))
    fp2.flush()

    sum_accuracy = 0
    for i in range(0, N_test, val_batchsize):
        val_x_batch = np.ndarray(
            (val_batchsize, 1, yoko, tate), dtype=np.float32)
        val_y_batch = np.ndarray((val_batchsize,), dtype=np.float32)
        val_batch_pool = [None] * val_batchsize

        for zz in range(val_batchsize):
            path, label = all_data[count]
            val_batch_pool[zz] = path
            val_x_batch[zz]=val_batch_pool[zz]
            val_y_batch[zz] = label
            count+=1
        x_batch = xp.asarray(val_x_batch)
        y_batch = xp.asarray(val_y_batch).reshape(batchsize,1)

        loss , ans = forward(x_batch, y_batch, train=False)
        sum_accuracy += float(loss.data) * len(val_y_batch)
        #pearson = scipy.stats.pearsonr(acc.data,ans.data)
        for sa in range(val_batchsize):
            gosa.write("%f %f" % (ans.data[sa],val_y_batch[sa]))
            gosa.write("\n")
            gosa.flush()


    count=0
    print "test mean squared error: %f" % (sum_accuracy / N_test)
    fp1.write("%d\t%f\n" % (epoch, sum_accuracy / N_test))
    fp1.flush()

end_time = time.clock()
print end_time - start_time

fp1.close()
fp2.close()
gosa.close()

import cPickle
# CPU環境でも学習済みモデルを読み込めるようにCPUに移してからダンプ
model.to_cpu()
cPickle.dump(model, open("model.pkl", "wb"), -1)
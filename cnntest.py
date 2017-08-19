#coding: utf-8
import numpy as np
import chainer
from chainer import cuda
import chainer.functions as F
from chainer import optimizers
import pylab
import matplotlib.pyplot as plt
import chainer
import chainer.links as L
import os
import scipy.stats
import cv2
import cPickle
from PIL import Image

gpu_flag = -1
height = 165
width = 1200

if gpu_flag >= 0:
    cuda.check_cuda_available()
xp = cuda.cupy if gpu_flag >= 0 else np


if gpu_flag >= 0:
    cuda.get_device(gpu_flag).use()
    model.to_gpu()
with open("model.pkl","rb") as i:
    model = cPickle.load(i)
 

def forward(x_data):
    x = chainer.Variable(x_data)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv1(x))), 3,stride=2)
    h = F.max_pooling_2d(F.relu(
        F.local_response_normalization(model.conv2(h))), 3,stride=2)
    h = F.relu(model.fc6(h))
    h = F.relu(model.fc7(h))
    h = model.fc8(h)
    #import pdb; pdb.set_trace()
    return F.softmax(h)

optimizer = optimizers.RMSpropGraves()
optimizer.setup(model)


def csvread(csvfile):
    
    pil = Image.fromarray(np.uint8(csvfile[0:width,0:height]))
    pil = pil.resize((255, 255))
    
    return np.asarray(pil).astype(np.float32).reshape((1,1,255,255))


def predict(csvfile):

    fishs = ["サケ","ブリ","イワシ","イカ","マグロ"]
    amount = ["0kg" ,"0kg" ,"0kg" ,"0kg" ,"0kg"]
    
    fishs_amount = []
    for i in range(5):
        x_batch = xp.asarray(csvread(csvfile))
        
        ans = forward(x_batch)
        fish = np.argmax(ans.data)#ソフトマックス関数で最も確率の高い漁獲量クラスを返す
    
        if fish==1:
            amount[i]="101~200kg"
        elif fish==2:
            amount[i]="201~300kg"
        elif fish==3:
            amount[i]="301~400kg"
        elif fish==4:
            amount[i]="700~kg"
        
        fishs_amount.append(fishs[i]+" "+amount[i])
    return fishs_amount

if __name__ == "__main__":
    csvfile = np.genfromtxt("2015_920.csv",delimiter=",",dtype=np.int32)


    print predict(csvfile)






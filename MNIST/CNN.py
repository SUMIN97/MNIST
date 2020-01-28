import numpy as np
import sys
import os
from array import array
import random
import math
import tqdm
from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm


#변수 정의
TrainImg = []
TrainLabel = []
TestImg = []
TestLabel = []
epoch = 0
Lost = 1000000000000000
PaddingNum = 0
TrainImgNum = 10000
KernelSize = 3
Filter1Size = 32
Filter2Size = 64
F1 = np.randn(KernelSize * KernelSize, Filter1Size) * np.sqrt(2/ KernelSize * KernelSize * Filter1Size)
Input = []
L1 = np.array()



def Padding(img, size):
    height, width = img.shape
    for i in range(height):
        for j in range(size): #size 횟수만큼 행의 맨 앞에 num을 추가
            img.insert(img[i], 0, PaddingNum)
        for j in range(size): #size 횟수만큼 행의 맨 뒤에 num을 추가
            img.append(img[i], PaddingNum)
    for j in range(size):
        img.insert(img, 0, np.zeros((width + size + size)))
        img.append(img, np.zeros((width + size + size)))
    return img

def ChangeImgToMatrix(img):
    height, width = img.shape
    matrix = np.array()
    for i in range(height-2):
        for j in range (width-2):
            array1 = img[i][j:j+KernelSize]
            array2 = img[i+1][j:j+KernelSize]
            array3 = img[i+2][j:j+KernelSize]
            matrix.append(np.concatenate([array1, array2, array3]))
        matrix.append()
    return matrix

def ReLU(img):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            img[i][j] = max(img[i][j], 0)
    return img

def MaxPooling(input, resultmatrix):
    depth, height, width = input.shape
    LayerMax = np.zeros(depth, height/2, width/2)
    for d in range(depth):
        for h in range(height):
            if h%2 != 0:
                continue
            else:
                for w in range(width):
                    if w %2 != 0:
                        continue
                    else:
                        list = [input[d][h][w],input[d][h][w+1], input[d][h+1][w], input[d][h+1][w+1]]
                        value = max(list)
                        index = list.index(value)
                        if index == 0:
                            resultmatrix[d][h][w] = 1
                        elif index == 1:
                            resultmatrix[d][h][w+1] = 1
                        elif index == 2:
                            resultmatrix[d][h+1][w] = 1
                        else:
                            resultmatrix[d][h+1][w+1] = 1

                        LayerMax[d][h/2][w/2] = value
    return LayerMax

#파일 읽기
fp_train_image = open('C:\\Users\\user\\Documents\\2019\\LAB\\MNIST\\training_set\\train-images.idx3-ubyte','rb')
fp_train_label = open('C:\\Users\\user\\Documents\\2019\\LAB\\MNIST\\training_set\\train-labels.idx1-ubyte', 'rb')
fp_test_image = open('C:\\Users\\user\\Documents\\2019\\LAB\\MNIST\\test_set\\t10k-images.idx3-ubyte','rb')
fp_test_label = open('C:\\Users\\user\\Documents\\2019\\LAB\\MNIST\\test_set\\t10k-labels.idx1-ubyte','rb')
#read mnist and show numberc

#train data 저장
while True:
    s  = fp_train_image.read(784) #784 바이트씩 읽음
    label = fp_train_label.read(1) #1 바이트 씩 읽음

    if not s:
        break
    if not label:
        break
    #unpack
    num = int(label[0])
    img = list(unpack(len(s) * 'B', s))  # byte를 unsigned char 형식으로
    TrainImg.append(np.reshape(img, (28,28)))
    TrainLabel.append(num)

TrainImg = np.array(TrainImg)
TrainImg = TrainImg/255.0
TrainLabel = np.array(TrainLabel)

for img in range(TrainImgNum):
    img = Padding(img, 1)
    input = ChangeImgToMatrix(img)
    L1 = np.matmul(input, F1)
    L1 = ReLU(L1)
    L1.T
    L1 = np.reshape(L1, (32, 28, 28))
    MaxPoolingL1ResultMatrix = np.zeros((32, 28, 28))
    L1Max = MaxPooling(L1, MaxPoolingL1ResultMatrix)







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

# 파일 읽기
fp_train_image = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\training_set\\train-images.idx3-ubyte', 'rb')
fp_train_label = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\training_set\\train-labels.idx1-ubyte', 'rb')
fp_test_image = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\test_set\\t10k-images.idx3-ubyte', 'rb')
fp_test_label = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\test_set\\t10k-labels.idx1-ubyte', 'rb')
# read mnist and show numberc
fp_train_image.read(16) #read first 16 byte
fp_train_label.read(8)  # 1바이트씩 읽음

fp_test_image.read(16)  # read first 16 byte
fp_test_label.read(8)  # 1바이트씩 읽음


# 변수 정의
TrainImg = []
TrainLabel = []
TestImg = []
TestLabel = []
epoch = 0
Lost = 1000000000000000
PaddingNum = 0
TrainImgNum = 10000
KernelSize = 3
Filter1Count = 4
Filter2Count = 8
F1 = np.random.randn(Filter1Count, KernelSize, KernelSize) / np.sqrt(KernelSize * KernelSize * Filter1Count)
F2 = np.random.randn(Filter2Count, KernelSize, KernelSize) / np.sqrt(KernelSize * KernelSize * Filter2Count)
F3 = np.random.randn(14 * 14 * Filter1Count, 10) / np.sqrt(14 * 14 * Filter1Count * 10)
Input = []
MaxPoolingL1Result = np.zeros((Filter1Count, 28, 28), dtype=float)
MaxPoolingL2Result = np.zeros((Filter2Count, 14, 14), dtype=float)
bias = np.ones((10))
learning_rate = 0.01
WrongCount = 0
plt.switch_backend('agg')
img = np.zeros((28,28)) #이미지가 저장될 부분
half_height = 14
half_width = 14


# train data 저장
while True:
    s = fp_train_image.read(784)  # 784 바이트씩 읽음
    label = fp_train_label.read(1)  # 1 바이트 씩 읽음

    if not s:
        break
    if not label:
        break
    # unpack
    num = int(label[0])
    img = np.reshape( unpack(len(s)*'B',s), (28,28))/255.0 # byte를 unsigned char 형식으로
    # img = list(unpack(len(s) * 'B', s))  # byte를 unsigned char 형식으로

    TrainImg.append(img)
    TrainLabel.append(num)

TrainImg = np.array(TrainImg)

# test data 저장
while True:
    s = fp_test_image.read(784)  # 784 바이트씩 읽음
    label = fp_test_label.read(1)  # 1 바이트 씩 읽음

    if not s:
        break
    if not label:
        break

    # unpack
    num = int(label[0])
    img = np.reshape( unpack(len(s)*'B',s), (28,28))/255.0  # byte를 unsigned char 형식으로

    TestImg.append(img)
    TestLabel.append(num)

TestImg = np.array(TestImg)


TrainImgNum = len(TrainImg)
TestImgNum = len(TestImg)
epoch = 0
while epoch < 20:
    # for i in tqdm.tqdm(range(1000)):
    for i in tqdm.tqdm(range(TrainImgNum)):
        L1 = np.copy(TrainImg[i])

        #Padding
        height, width = L1.shape
        L1 = np.insert(L1, [0, width], [PaddingNum, PaddingNum], axis=1)
        L1 = np.insert(L1, 0, np.zeros(width + 2), axis=0)
        L1Padding = np.insert(L1, height + 1, np.zeros(width + 2), axis=0)

        #Convolution
        L1ConvolAfter = np.zeros((Filter1Count, 28, 28))
        for f in range(Filter1Count):
            for h in range(28):
                for w in range(28):
                    arr1 = np.reshape(L1Padding[h:h+3, w:w+3], (1,-1))
                    arr2 = np.reshape(F1[f:f+1, :, :], (1, -1))
                    L1ConvolAfter[f][h][w] = np.sum(arr1 * arr2)

        #ReLU
        for d in range(L1ConvolAfter.shape[0]):
            for h in range(L1ConvolAfter.shape[1]):
                for w in range(L1ConvolAfter.shape[2]):
                    L1ConvolAfter[d][h][w] = np.max(np.array([0, L1ConvolAfter[d][h][w]]))

        #MaxPooling
        MaxPoolingL1Result = np.zeros((Filter1Count, 28, 28))
        L2 = np.zeros((Filter1Count, 14, 14))

        for d in range(Filter1Count):
            for h in range(half_height):
                for w in range(half_width):
                    arr = np.reshape(L1ConvolAfter[d:d+1, 2*h:2*h+2, 2*w:2*w+2], (1, -1))
                    L2[d][h][w] = np.max(arr)
                    index = np.argmax(arr)
                    MaxPoolingL1Result[d][2*h +int(index/2)][2*w + index%2] = 1
#******************************************************************************************************

        #Padding
        height, width = L2.shape
        L2 = np.insert(L1, [0, width], [PaddingNum, PaddingNum], axis=1)
        L2 = np.insert(L1, 0, np.zeros(width + 2), axis=0)
        L2Padding = np.insert(L2, height + 1, np.zeros(width + 2), axis=0)

        #Convolution
        L2ConvolAfter = np.zeros((Filter2Count, 14, 14))
        for f in range(Filter2Count):
            for h in range(14):
                for w in range(14):
                    arr1 = np.reshape(L2Padding[h:h+3, w:w+3], (1,-1))
                    arr2 = np.reshape(F2[f:f+1, :, :], (1, -1))
                    L2ConvolAfter[f][h][w] = np.sum(arr1 * arr2)

        #ReLU
        for d in range(L2ConvolAfter.shape[0]):
            for h in range(L2ConvolAfter.shape[1]):
                for w in range(L2ConvolAfter.shape[2]):
                    L2ConvolAfter[d][h][w] = np.max(np.array([0, L2ConvolAfter[d][h][w]]))

        #MaxPooling
        MaxPoolingL2Result = np.zeros((Filter2Count, 14, 14))
        L3 = np.zeros((Filter2Count, 7, 7))

        for d in range(Filter2Count):
            for h in range(7):
                for w in range(7):
                    arr = np.reshape(L2ConvolAfter[d:d+1, 2*h:2*h+2, 2*w:2*w+2], (1, -1))
                    L3[d][h][w] = np.max(arr)
                    index = np.argmax(arr)
                    MaxPoolingL2Result[d][2*h +int(index/2)][2*w + index%2] = 1

        L3Reshape = np.reshape(L3, (1, -1))
        L3Input = np.matmul(L3Reshape, F3) + bias
        # L3Input = L3Input - np.max(L3Input)
        L3Output = 1 / (1 + np.exp(-L3Input))


        # Backpropagation
        onehot = np.zeros(10, dtype=float)
        onehot[TrainLabel[i]] = 1

        # DErrorByL3Output =P - target
        DErrorByL3Input = L3Output - onehot
        DErrorByBias = DErrorByL3Input
        DErrorByF3 = np.matmul(L2Reshape.T, DErrorByL3Input)
        DErrorByL3Reshape = np.matmul(DErrorByL3Input, F3.T)
        DErrorByL3 = np.reshape(DErrorByL3Reshape, (Filter2Count, 7, 7))

        DErrorByL2ConvolAfter = MaxPoolingL2Result

        for d in range(Filter2Count):
            for h in range(7):
                for w in range(7):
                    DErrorByL2ConvolAfter[d:d+1, 2*h:2*h+2, 2*w:2*w+2] *= DErrorByL3[d][h][w]

        #Backpropagate Convol
        DErrorByF2 = np.zeros((Filter2Count, KernelSize, KernelSize))
        for d in range(Filter2Count):
            for h in range(KernelSize):
                for w in range(KernelSize):
                    DErrorByF2[d][h][w] = np.sum(L1Padding[h:h+14, w:w+14] * DErrorByL2ConvolAfter[d:d+1, :, :])



        DErrorByL1ConvolAfter = MaxPoolingL1Result

        for d in range(Filter1Count):
            for h in range(half_height):
                for w in range(half_width):
                    DErrorByL1ConvolAfter[d:d+1, 2*h:2*h+2, 2*w:2*w+2] *= DErrorByL2[d][h][w]

        #Backpropagate Convol
        DErrorByF1 = np.zeros((Filter1Count, KernelSize, KernelSize))
        for d in range(Filter1Count):
            for h in range(KernelSize):
                for w in range(KernelSize):
                    DErrorByF1[d][h][w] = np.sum(L1Padding[h:h+28, w:w+28] * DErrorByL1ConvolAfter[d:d+1, :, :])


        F1 = F1 - learning_rate * DErrorByF1
        F3 = F3 - learning_rate * DErrorByF3
        bias = bias - learning_rate *DErrorByBias

    #test
    for i in tqdm.tqdm(range(TestImgNum)):
        L1 = np.copy(TestImg[i])

        #Padding
        height, width = L1.shape
        L1 = np.insert(L1, [0, width], [PaddingNum, PaddingNum], axis=1)
        L1 = np.insert(L1, 0, np.zeros(width + 2), axis=0)
        L1Padding = np.insert(L1, height + 1, np.zeros(width + 2), axis=0)

        #Convolution
        L1ConvolAfter = np.zeros((Filter1Count, 28, 28))
        for f in range(Filter1Count):
            for h in range(28):
                for w in range(28):
                    arr1 = np.reshape(L1Padding[h:h+3, w:w+3], (1,-1))
                    arr2 = np.reshape(F1[f:f+1, :, :], (1, -1))
                    L1ConvolAfter[f][h][w] = np.sum(arr1 * arr2)

        #ReLU
        for d in range(L1ConvolAfter.shape[0]):
            for h in range(L1ConvolAfter.shape[1]):
                for w in range(L1ConvolAfter.shape[2]):
                    L1ConvolAfter[d][h][w] = np.max(np.array([0, L1ConvolAfter[d][h][w]]))

        #MaxPooling
        MaxPoolingL1Result = np.zeros((Filter1Count, 28, 28))
        L2 = np.zeros((Filter1Count, 14, 14))

        for d in range(Filter1Count):
            for h in range(half_height):
                for w in range(half_width):
                    arr = np.reshape(L1ConvolAfter[d:d+1, 2*h:2*h+2, 2*w:2*w+2], (1, -1))
                    L2[d][h][w] = np.max(arr)
                    index = np.argmax(arr)
                    MaxPoolingL1Result[d][2*h +int(index/2)][2*w + index%2] = 1

        L2Reshape = np.reshape(L2, (1, -1))
        L3Input = np.matmul(L2Reshape, F3) + bias
        # L3Input = L3Input - np.max(L3Input)
        L3Output = 1 / (1 + np.exp(-L3Input))

        if ((np.argmax(L3Output)) != TestLabel[i]):
            WrongCount+=1

    print("Wrong count: ", WrongCount, "\n")
    print("Correct Percent: ", (TestImgNum - WrongCount)/TestImgNum * 100)
    epoch += 1
    WrongCount = 0













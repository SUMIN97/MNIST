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
F1 = np.random.randn(KernelSize * KernelSize, Filter1Count) / np.sqrt(KernelSize * KernelSize * Filter1Count)
# F2 = np.random.randn(Filter1Count, KernelSize * KernelSize, Filter2Count) * np.sqrt(
#     2 / Filter1Count * (KernelSize * KernelSize) * Filter2Count)
F3 = np.random.randn(14 * 14 * Filter1Count, 10) / np.sqrt(14 * 14 * Filter1Count * 10)
Input = []
MaxPoolingL1Result = np.zeros((Filter1Count, 28, 28), dtype=float)
MaxPoolingL2Result = np.zeros((Filter2Count, 14, 14), dtype=float)
bias = np.ones((10))
learning_rate = 0.01
WrongCount = 0
plt.switch_backend('agg')


def Padding(img, size):
    if np.array(img.shape).size == 2:
        height, width = img.shape
        img = np.insert(img, [0, width], [PaddingNum, PaddingNum], axis=1)
        img = np.insert(img, 0, np.zeros(width + 2), axis=0)
        img = np.insert(img, height + 1, np.zeros(width + 2), axis=0)
        return img
    else:
        depth, height, width = img.shape
        result = np.zeros((depth, height + size * 2, width + size * 2))
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    result[d][h + 1][w + 1] = img[d][h][w]
        return result


def ChangeToConvolutionMatrix(img):
    if np.array(img.shape).size == 2:
        height, width = img.shape
        matrix = np.zeros(KernelSize * KernelSize)
        for i in range(28):
            for j in range(28):
                array = img[i][j:j + KernelSize]
                for k in np.arange(1, KernelSize):
                    array1 = img[i + k][j:j + KernelSize].copy()
                    array = np.concatenate((array, array1))
                matrix = np.vstack((matrix, array))
        matrix = np.delete(matrix, 0, axis=0)
        return matrix
    else:
        depth, height, width = img.shape
        cube = np.zeros((depth, 14 * 14, 9))

        for d in range(depth):
            matrix = np.zeros(KernelSize * KernelSize)
            for i in range(14):
                for j in range(14):
                    array = img[d][i][j:j + KernelSize]
                    for k in np.arange(1, KernelSize):
                        array1 = img[d][i + k][j:j + KernelSize]
                        array = np.concatenate((array, array1))
                    matrix = np.vstack((matrix, array))
            matrix = np.delete(matrix, 0, axis=0)
            cube[d] = matrix
        return cube


def ReLU(img):
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            img[i][j] = max(img[i][j], 0)
    return img


def MaxPooling(input):
    depth, height, width = input.shape
    LayerMax = np.zeros((depth, int(height / 2), int(width / 2)))
    for d in range(depth):
        for h in range(height):
            if h % 2 != 0:
                continue
            else:
                for w in range(width):
                    if w % 2 != 0:
                        continue
                    else:
                        list = [input[d][h][w], input[d][h][w + 1], input[d][h + 1][w], input[d][h + 1][w + 1]]
                        value = max(list)
                        index = list.index(value)
                        if index == 0:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h][w] = 1
                            elif depth == Filter2Count:
                                MaxPoolingL2Result[d][h][w] = 1
                            else:
                                print("MaxPooling Error!\n")
                                return -1

                        elif index == 1:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h][w + 1] = 1
                            elif depth == Filter2Count:
                                MaxPoolingL2Result[d][h][w + 1] = 1
                            else:
                                print("MaxPooling Error!\n")
                                return -1

                        elif index == 2:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h + 1][w] = 1
                            elif depth == Filter2Count:
                                MaxPoolingL2Result[d][h + 1][w] = 1
                            else:
                                print("MaxPooling Error!\n")
                                return -1
                        else:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h + 1][w + 1] = 1
                            elif depth == Filter2Count:
                                MaxPoolingL2Result[d][h + 1][w + 1] = 1
                            else:
                                print("MaxPooling Error!\n")
                                return -1

                        LayerMax[d][int(h / 2)][int(w / 2)] = value
    return LayerMax


def Convolution(img):
    # F1 일때
    if np.array(img.shape).size == 2:
        result = np.matmul(img, F1)
        return result
    # F2 일때
    else:
        depth, height, width = img.shape
        Filterdepth, Filterheight, Filterwidth = F2.shape
        result = np.zeros((height, Filterwidth))
        for h in range(height):
            for w in range(Filterwidth):
                sum = 0
                for d in range(depth):
                    array1 = img[d][h]
                    array2 = F2[d].T[w]
                    sum = sum + np.sum(array1 * array2)
                result[h][w] = sum
        return result


def Softmax(input):
    layer = input - np.max(input)
    layer = np.exp(layer)
    sum = np.sum(layer)
    layer = layer / sum
    return layer


def BackpropagateMaxPooling(input):
    result = np.copy(MaxPoolingL1Result)
    depth, height, width = MaxPoolingL1Result.shape

    half_height = int(height/2)
    half_width = int(width/2)
    for d in range(depth):
        for h in range(half_height):
            for w in range(half_width):
                MaxPoolingL1Result[d][2*h: 2*h+2][2 * w : 2 *w +2] *= input[d][h][w]
    return result


# 파일 읽기
fp_train_image = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\training_set\\train-images.idx3-ubyte', 'rb')
fp_train_label = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\training_set\\train-labels.idx1-ubyte', 'rb')
fp_test_image = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\test_set\\t10k-images.idx3-ubyte', 'rb')
fp_test_label = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\test_set\\t10k-labels.idx1-ubyte', 'rb')
# read mnist and show numberc

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
    img = np.array(unpack(len(s) * 'B', s))  # byte를 unsigned char 형식으로
    # img = list(unpack(len(s) * 'B', s))  # byte를 unsigned char 형식으로
    if len(img) != 784:
        continue
    elif num > 9:
        continue
    else:
        img = np.reshape(img, (28, 28))
        img = img / 255.0

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
    img = np.array(unpack(len(s) * 'B', s))  # byte를 unsigned char 형식으로

    if len(img) != 784:
        continue
    elif num > 9:
        continue
    else:
        img = np.reshape(img, (28,28))
        img = img / 255.0
        TestImg.append(img)
        TestLabel.append(num)

TestImg = np.array(TestImg)


TrainImgNum = len(TrainImg)
TestImgNum = len(TestImg)
epoch = 0
while epoch < 10:
    # for i in tqdm.tqdm(range(20000)):
    for i in tqdm.tqdm(range(TrainImgNum)):
        L1 = np.copy(TrainImg[i])
        height, width = L1.shape
        L1 = np.insert(L1, [0, width], [PaddingNum, PaddingNum], axis=1)
        L1 = np.insert(L1, 0, np.zeros(width + 2), axis=0)
        L1Padding = np.insert(L1, height + 1, np.zeros(width + 2), axis=0)

        L1ConvolBefore = ChangeToConvolutionMatrix(L1Padding)
        L1ConvolAfter = np.matmul(L1ConvolBefore, F1)
        # L1ConvolAfter = Convolution(L1ConvolBefore)
        L1ReLu = ReLU(L1ConvolAfter)
        L1T = L1ReLu.T
        L1Reshape = np.reshape(L1T, (Filter1Count, 28, 28))

        MaxPoolingL1Result = np.zeros((Filter1Count, 28, 28))
        L2 = MaxPooling(L1Reshape)

        L2Reshape = np.reshape(L2, (1, -1))
        L3Input = np.matmul(L2Reshape, F3) + bias
        # L3Input = L3Input - np.max(L3Input)
        L3Output = 1 / (1 + np.exp(-L3Input))


        # Backpropagation
        onehot = np.zeros(10, dtype=float)
        onehot[TrainLabel[i]] = 1

        # DErrorByL3Output =P - target
        DErrorByL3Input = L3Output - onehot
        DErrorByBias = DErrorByL3Input
        DErrorByF3 = np.matmul(L2Reshape.T, DErrorByL3Input)
        DErrorByL2Reshape = np.matmul(DErrorByL3Input, F3.T)
        DErrorByL2 = np.reshape(DErrorByL2Reshape, (Filter1Count, 14, 14))
        DErrorByL1Reshape = BackpropagateMaxPooling(DErrorByL2)
        DErrorByL1T = np.reshape(DErrorByL1Reshape, (Filter1Count, 28*28))
        DErrorByL1ReLu = DErrorByL1T.T

        # ReLU
        DErrorByL1ConvolAfter = DErrorByL1ReLu

        # F1
        DErrorByF1 = np.matmul(L1ConvolBefore.T, DErrorByL1ConvolAfter)
        DErrorByL1ConvolBefore = np.matmul(DErrorByL1ConvolAfter, F1.T)

        F1 = F1 - learning_rate * DErrorByF1
        F3 = F3 - learning_rate * DErrorByF3
        bias = bias - learning_rate *DErrorByBias

    #test
    for i in tqdm.tqdm(range(TestImgNum)):
        L1 = TestImg[i]
        L1Padding = Padding(L1, 1)
        L1ConvolBefore = ChangeToConvolutionMatrix(L1Padding)
        L1ConvolAfter = Convolution(L1ConvolBefore)
        L1ReLu = ReLU(L1ConvolAfter)
        L1T = L1ReLu.T
        L1Reshape = np.reshape(L1T, (Filter1Count, 28, 28))
        MaxPoolingL1Result = np.zeros((Filter1Count, 28, 28))
        L2 = MaxPooling(L1Reshape)

        L2Reshape = np.reshape(L2, (1, -1))
        L3Input = np.matmul(L2Reshape, F3) + bias
        L3Output = 1 / (1 + np.exp(-L3Input))

        if ((np.argmax(L3Output) + 1) != TestLabel[i]):
            WrongCount+=1

    print("Wrong count: ", WrongCount, "\n")
    epoch += 1
    WrongCount = 0













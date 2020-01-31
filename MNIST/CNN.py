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
Filter1Count = 4
Filter2Count = 8
F1 = np.random.randn(KernelSize * KernelSize, Filter1Count) * np.sqrt(2/ KernelSize * KernelSize * Filter1Count)
F2 = np.random.randn(Filter1Count, KernelSize * KernelSize, Filter2Count) * np.sqrt(2/Filter1Count * (KernelSize * KernelSize) * Filter2Count)
F3 = np.random.randn(7 *7 *Filter2Count, 10) * np.sqrt(2 / 7 * 7 * Filter2Count * 10)
Input = []
MaxPoolingL1Result = np.zeros((Filter1Count, 28, 28))
MaxPoolingL2Result = np.zeros((Filter2Count, 14, 14))
bias = np.ones((10))
learning_rate = 0.01

plt.switch_backend('agg')


def Padding(img, size):
    if np.array(img.shape).size == 2:
        height, width = img.shape
        img = np.insert(img, [0, width], [PaddingNum, PaddingNum], axis=1)
        img = np.insert(img, 0, np.zeros(width+2), axis=0)
        img = np.insert(img, height+1, np.zeros(width + 2), axis=0)
        return img
    else:
        depth, height, width = img.shape
        result = np.zeros((depth, height + size*2, width + size*2))
        for d in range(depth):
            for h in range(height):
                for w in range(width):
                    result[d][h+1][w+1] = img[d][h][w]
        return result



def ChangeToConvolutionMatrix(img):
    if np.array(img.shape).size == 2:
        height, width = img.shape
        matrix = np.zeros(KernelSize * KernelSize)
        for i in range(28):
            for j in range (28):
                array = img[i][j:j+KernelSize]
                for k in np.arange(1, KernelSize):
                    array1 = img[i + k][j:j+KernelSize]
                    array = np.concatenate((array, array1))
                matrix = np.vstack((matrix, array))
        matrix = np.delete(matrix, 0, axis = 0)
        return matrix
    else:
        depth, height, width = img.shape
        cube = np.zeros((depth, 14*14, 9))

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
    LayerMax = np.zeros((depth, int(height/2), int(width/2)))
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
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h][w] = 1
                            elif depth == Filter2Count:
                                MaxPoolingL2Result[d][h][w] = 1
                            else:
                                print("MaxPooling Error!\n")
                                return -1

                        elif index == 1:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h][w+1] = 1
                            elif depth == Filter2Count:
                                MaxPoolingL2Result[d][h][w+1] = 1
                            else:
                                print("MaxPooling Error!\n")
                                return -1

                        elif index == 2:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h+1][w] = 1
                            elif depth == Filter2Count:
                                MaxPoolingL2Result[d][h+1][w] = 1
                            else:
                                print("MaxPooling Error!\n")
                                return -1
                        else:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h+1][w+1] = 1
                            elif depth == Filter2Count:
                                MaxPoolingL2Result[d][h+1][w+1] = 1
                            else:
                                print("MaxPooling Error!\n")
                                return -1

                        LayerMax[d][int(h/2)][int(w/2)] = value
    return LayerMax

def Convolution(img):
    #F1 일때
    if np.array(img.shape).size == 2:
        result = np.matmul(img, F1)
        return result
    #F2 일때
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
    layer = layer/sum
    return layer

def BackprpagateMaxPooling(input, resultmatrix):
    depth, height, width = resultmatrix.shape
    result = resultmatrix
    for d in range(depth):
        for h in range(height):
            for w in range(width):
                if result[d][h][w] == 0:
                    continue
                else:
                    result[d][h][w] = result[d][h][w] * input[d][int(h/2)][int(w/2)]
    return result





#파일 읽기
fp_train_image = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\training_set\\train-images.idx3-ubyte','rb')
fp_train_label = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\training_set\\train-labels.idx1-ubyte', 'rb')
fp_test_image = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\test_set\\t10k-images.idx3-ubyte','rb')
fp_test_label = open('C:\\Users\\user\\Documents\\LAB\\MNIST\\test_set\\t10k-labels.idx1-ubyte','rb')
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
    if len(img) !=  784:
        continue
    else:
        img = np.reshape(img, (28,28))
        img = img/255.0
        TrainImg.append(list(img))
        TrainLabel.append(num)

TrainImg = np.array(TrainImg)
TrainImgNum = len(TrainImg)

for i in range(TrainImgNum):
    L1 = TrainImg[i]
    L1Padding = Padding(L1, 1)
    L1ConvolBefore = ChangeToConvolutionMatrix(L1Padding)
    L1ConvolAfter = Convolution(L1ConvolBefore)
    L1ReLu = ReLU(L1ConvolAfter)
    L1T = L1ReLu.T
    L1Reshape = np.reshape(L1T, (Filter1Count, 28, 28))
    MaxPoolingL1Result = np.zeros((Filter1Count, 28, 28))
    L2 = MaxPooling(L1Reshape)

    L2Padding = Padding(L2, 1)
    L2ConvolBefore = ChangeToConvolutionMatrix(L2Padding)
    L2ConvolAfter = Convolution(L2ConvolBefore)
    L2ReLU = ReLU(L2ConvolAfter)
    L2T = L2ReLU.T
    L2ReShape = np.reshape(L2T, (Filter2Count, 14, 14))
    MaxPoolingL2ResultMatrix = np.zeros((Filter2Count, 14, 14))
    L3 = MaxPooling(L2ReShape)
    L3Reshape = np.reshape(L3, (1, -1))
    L3FullConnect = np.matmul(L3Reshape, F3)
    L3Output = L3FullConnect + bias
    L3Output  = L3Output - np.max(L3Output)
    L3Exp = np.exp(L3Output)
    sum = np.sum(L3Output)
    P = L3Exp/sum

    # Y = Softmax(L3Output)
    #cross entropy

    #Backpropagation
    target = np.zeros((10))
    DErrorByL3Output =P - target
    DErrorByBias = DErrorByL3FullConnect = DErrorByL3Output
    DErrorByF3 = np.matmul(L3.Reshape.T , DErrorByL3FullConnect)
    DErrorByL3Reshape = np.matmul(DErrorByL3FullConnect, F3.T)
    DErrorByL3 = np.reshape(DErrorByL3Reshape, (Filter2Count, 7, 7))
    DErrorByL2Reshape = BackprpagateMaxPooling(DErrorByL3, MaxPoolingL2ResultMatrix)

    DErrorByL2T = DErrorByL2Reshape.T
    #ReLus는 동일하니까
    DErrorByL2ConvolAfter = DErrorByL2T

    #F2
    DErrorByF2 = np.zeros((Filter1Count, KernelSize * KernelSize, Filter2Count))
    DErrorByL2ConvolBefore = np.zeros((Filter1Count, 14 * 14, KernelSize*KernelSize))

    for h in range(DErrorByL2ConvolAfter.shape[0]): #14*14
        for w in range(DErrorByL2ConvolAfter.shape[1]): #Filter2Count
            for d in range(Filter1Count):
                for k in range(KernelSize * KernelSize):
                    DErrorByL2ConvolBefore[d][h][k] = F2[d][k][w] * DErrorByL2ConvolAfter[h][w]
                    DErrorByF2[d][k][w] = L2ConvolBefore[d][h][k] * DErrorByL2ConvolAfter[h][w]

    #Padding & ChangeToConvolutionMatrix Backpropagation
    DErrorByL2Padding = np.zeros(Filter1Count, 16, 16)


 """   
    img = Padding(TrainImg[i], 1)
    input = ChangeToConvolutionMatrix(img)
    L1 = Convolution(input, F1)
    L1 = ReLU(L1)
    L1.T
    L1 = np.reshape(L1, (Filter1Count, 28, 28))
    MaxPoolingL1ResultMatrix = np.zeros((Filter1Count, 28, 28))
    L1Max = MaxPooling(L1)
    L2input = Padding(L1Max, 1) #결과가 (Filter1Count, 14*14+2, 9+2)
    L2input = ChangeToConvolutionMatrix(L2input)
    L2 = Convolution(L2input, F2)
    L2 = ReLU(L2)
    L2.T
    L2 = np.reshape(L2, (Filter2Count, 14, 14))
    MaxPoolingL2ResultMatrix = np.zeros((Filter2Count, 14, 14))
    L2Max = MaxPooling(L2)
    L3input = np.reshape(L2Max, (1, -1))
    L3input = np.matmul(L3input, F3)
    L3input = L3input + bias
    L3 = Softmax(L3input)

    #Backpropagation
    target = np.zeros((10))
    # target = np.array([0.001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.0001])
    target[TrainLabel[i]] = 1
    # DErrorByL3 = (L3 - target)/L3 * (1 - L3)
    DErrorByL3input = L3 - target
    DErrorByBias = DErrorByL3input * 1
    DErrorbyF3 = np.matmul(L3input.T, DErrorByBias)
    DErrorByL3input = np.matmul(DErrorByBias, F3.T)
    DErrorByL2Max = np.reshape(DErrorByL3input, (Filter2Count, 7, 7))
    DErrorByL2 = BackprpagateMaxPooling(DErrorByL2Max, MaxPoolingL2ResultMatrix)
    DErrorByL2 = np.reshape(DErrorByL2, (Filter2Count, 14 *14))
    DErrorByL2.T

    #F2
    DErrorByF2 = np.zeros((Filter1Count, KernelSize * KernelSize, Filter2Count))
    DErrorByL1MaxPadding = np.zeros((Filter1Count, 14 * 14, KernelSize*KernelSize))

    for h in range(DErrorByL2.shape[0]): #14*14
        for w in range(DErrorByL2.shape[1]): #Filter2Count
            for d in range(Filter1Count):
                for k in range(KernelSize * KernelSize):
                    DErrorByL1MaxPadding[d][h][k] = F2[d][k][w] * DErrorByL2[h][w]
                    DErrorByF2[d][k][w] = L1Max[d][h][k] * DErrorByL2[h][w]

    #Padding & ChangeToConvolutionMatrix Backpropagation
    DErrorByL1Max = np.zeros(Filter1Count, 14, 14)


    DErrorByL1Max = np.reshape(DErrorByL1Max, (Filter1Count, 14, 14))
    DErrorByL1 = BackprpagateMaxPooling(DErrorByL1Max, MaxPoolingL1ResultMatrix)
    DErrorByL1 = np.reshape(DErrorByL1, (Filter1Count, 28 * 28))
    DErrorByL1.T

    #F1
    DErrorByF1 = np.matmul(img.T, DErrorByL1)

    F1 = F1 - learning_rate * DErrorByF1
    F2 = F2 - learning_rate *DErrorByF2
    F3 = F3 - learning_rate *DErrorbyF3

"""




















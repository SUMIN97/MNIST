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

#파일 읽기
fp_train_image = open('C:\\Users\\user\\Documents\\2019\\LAB\\MNIST\\training_set\\train-images.idx3-ubyte','rb')
fp_train_label = open('C:\\Users\\user\\Documents\\2019\\LAB\\MNIST\\training_set\\train-labels.idx1-ubyte', 'rb')
fp_test_image = open('C:\\Users\\user\\Documents\\2019\\LAB\\MNIST\\test_set\\t10k-images.idx3-ubyte','rb')
fp_test_label = open('C:\\Users\\user\\Documents\\2019\\LAB\\MNIST\\test_set\\t10k-labels.idx1-ubyte','rb')
#read mnist and show number

#사용할 변수 초기화
L0NodeNum = 784
L1NodeNum = int(input('첫 번째 Hidden Layer Node 갯수를 입력하세요:'))
L2NodeNum = int(input('두 번째 Hidden Layer Node 갯수를 입력하세요:'))
L3NodeNum =10 #L3가 Output Lyaer


TrainImg = []
TrainLabel = []
TestImg = []
TestLabel = []
epoch = 0
Lost = 1000000000000000
learning_rate = 0.01
count=0
Error_image_count=0
TotalTestImg = 0
index=0
Batch_Size = 100


#W 의 초깃값이 표준편차가 작은 분표를 갖도록 초기화
W0 = np.random.randn(L0NodeNum, L1NodeNum)/math.sqrt(L0NodeNum)
W1 = np.random.randn(L1NodeNum, L2NodeNum) /math.sqrt(L1NodeNum)
W2 = np.random.randn(L2NodeNum, L3NodeNum)/math.sqrt(L2NodeNum)

Layer1Input = np.zeros(L1NodeNum, dtype=float)
Layer1Output = np.zeros(L1NodeNum, dtype=float)
Layer2Input = np.zeros(L2NodeNum, dtype=float)
Layer2Output = np.zeros(L2NodeNum, dtype=float)
Layer3Input = np.zeros(L3NodeNum, dtype=float)
Layer3Output = np.zeros(L3NodeNum, dtype=float)

TestLayer1Input = np.zeros(L1NodeNum, dtype=float)
TestLayer1Output = np.zeros(L1NodeNum, dtype=float)
TestLayer2Input = np.zeros(L2NodeNum, dtype=float)
TestLayer2Output = np.zeros(L2NodeNum, dtype=float)
TestLayer3Input = np.zeros(L3NodeNum, dtype=float)
TestLayer3Output = np.zeros(L3NodeNum, dtype=float)

BiasL1 = [1.0] * L1NodeNum
BiasL2 = [1.0] * L2NodeNum
BiasL3 = [1.0] * L3NodeNum

DLostWithLayer3Input = np.zeros(L3NodeNum, dtype=float)
DLostWithLayer2Input = np.zeros(L2NodeNum, dtype=float)
DLostWithLayer1Input = np.zeros(L1NodeNum, dtype=float)

DLostWithW0 = np.zeros((L0NodeNum, L1NodeNum), dtype=float)
DLostWithW1 = np.zeros((L1NodeNum, L2NodeNum), dtype=float)
DLostWithW2 = np.zeros((L2NodeNum, L3NodeNum), dtype=float)

DLostWithBiasL3 = np.zeros(L3NodeNum, dtype=float)
DLostWithBiasL2 = np.zeros(L2NodeNum, dtype=float)
DLostWithBiasL1 = np.zeros(L1NodeNum, dtype=float)

s  = fp_train_image.read(16) #read first 16 byte
label = fp_train_label.read(8)  # 1바이트씩 읽음

TEST_IMAGE = fp_test_image.read(16)  # read first 16 byte
TEST_INDEX = fp_test_label.read(8)  # 1바이트씩 읽음


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

    TrainImg.append(img)
    TrainLabel.append(num)

TrainImg = np.array(TrainImg)
TrainImg = TrainImg/255.0
TrainLabel = np.array(TrainLabel)



#test data 저장
while True:
    s = fp_test_image.read(784)  # 784 바이트씩 읽음
    label = fp_test_label.read(1)  # 1 바이트 씩 읽음

    if not s:
        break
    if not label:
        break

    # unpack
    num = int(label[0])
    img =  np.array(unpack(len(s) * 'B', s))  # byte를 unsigned char 형식으로
    TestImg.append(img)
    TestLabel.append(num)

TestImg = np.array(TestImg)
TestImg = TestImg/255.0
TestLabel = np.array(TestLabel)

TotalTrainImg = len(TrainImg)
TotalTestImg = len(TestImg)

#epoch 횟수 지정
while epoch < 20:
    Error_image_count = 0
    """
    SampleIndex = np.random.choice(TotalTrainImg, Batch_Size, replace=False)
    SampleImg = TrainImg[SampleIndex]
    SampleLabel = TrainLabel[SampleIndex]
    """
# batch size = 100
    for i in tqdm.tqdm(range(len(TrainImg))):
        img = TrainImg[i]
        label = TrainLabel.item(i)
        onehot = np.zeros(10, dtype=float)
        onehot[label] = 1

        L1NodeSum = 0.0
        #Layer1

        for index1 in range(L1NodeNum):
            sum = 0.0
            #784개의 모든 픽셀
            for index0 in range(L0NodeNum):
                sum += img[index0] * W0[index0][index1]
            #bias
            sum += BiasL1[index1]
            Layer1Input[index1] = sum

        #sigmoid 사용
        Layer1Output = 1 / (1 + np.exp(-Layer1Input))

        #print("#######################Layer2####################")
        #Layer2
        L2NodeSum = 0.0
        for index2 in range(L2NodeNum):
            sum = 0.0
            for index1 in range(L1NodeNum):
                sum += Layer1Output[index1] * W1[index1][index2]
            sum += BiasL2[index2]
            Layer2Input[index2] = sum

        Layer2Output = 1 / (1 + np.exp(-Layer2Input))

        #print("#######################Layer3####################")
        # Layer3
        L3NodeSum = 0.0
        for index3 in range(L3NodeNum):
            sum = 0.0
            for index2 in range(L2NodeNum):
                sum += Layer2Output[index2] * W2[index2][index3]
            sum += BiasL3[index3]
            Layer3Input[index3] = sum

        #softmax
        #cross-entropy 사용 후 backpropagation 과정
        DLostWithLayer3Input = (Layer3Input - onehot)

        #DLostWithW2
        for index2 in range(L2NodeNum):
            for index3 in range(L3NodeNum):
                DLostWithW2[index2][index3] = Layer2Output[index2] * DLostWithLayer3Input[index3]


        # DLostWithBiasL3
        DLostWithBiasL3 = DLostWithLayer3Input * 1.0

        # DLostWithLayer2Input
        for index2 in range(L2NodeNum):
            DLostWithLayer2Input[index2] = Layer2Output[index2] * (1 - Layer2Output[index2])
            sum = 0.0
            for index3 in range(L3NodeNum):
                sum +=  DLostWithLayer3Input[index3] * W2[index2][index3]
            DLostWithLayer2Input[index2] *= sum

        #DLostWithW1
        for index1 in range(L1NodeNum):
            for index2 in range(L2NodeNum):
                DLostWithW1[index1][index2] = Layer1Output[index1] * DLostWithLayer2Input[index2]
        #DLostWithBiasL2
        DLostWithBiasL2 = DLostWithLayer2Input * 1.0

        #DLostWithLayer1Input
        for index1 in range(L1NodeNum):
            DLostWithLayer1Input[index1] = Layer1Output[index1] * (1 - Layer1Output[index1])
            sum = 0.0
            for index2 in range(L2NodeNum):
                sum += DLostWithLayer2Input[index2] * W1[index1][index2]
            DLostWithLayer1Input[index1] *= sum

        #DLostWithW0
        for index0 in range(L0NodeNum):
            for index1 in range(L1NodeNum):
                DLostWithW0[index0][index1] = img[index0] * DLostWithLayer1Input[index1]

        # DLostWithBiasL1
        DLostWithBiasL1 = DLostWithLayer1Input * 1.0

        W0 = W0 - learning_rate * DLostWithW0
        #if count % 100 == 0:
            #print("W0[512][0]:", W0[512][0]);
        W1 = W1 - learning_rate * DLostWithW1
        W2 = W2 - learning_rate * DLostWithW2

        BiasL1 = BiasL1 - learning_rate * DLostWithBiasL1
        BiasL2 = BiasL2 - learning_rate * DLostWithBiasL2
        BiasL3 = BiasL3 - learning_rate * DLostWithBiasL3

    ####################test data#####################
    # epoch 마다
    Error_image_count = 0
    # classifiedImg 마다
    for i in tqdm.tqdm(range(TotalTestImg)):
        img = TestImg[i]
        label = TestLabel[i]

        # Layer1
        for index1 in range(L1NodeNum):
            sum = 0.0
            for index0 in range(L0NodeNum):
                sum += img[index0] * W0[index0][index1]
            # bias
            sum += BiasL1[index1]
            TestLayer1Input[index1] = sum

        TestLayer1Output = 1 / (1 + np.exp(-TestLayer1Input))

        # Layer2
        for index2 in range(L2NodeNum):
            sum = 0.0
            for index1 in range(L1NodeNum):
                sum += TestLayer1Output[index1] * W1[index1][index2]
            sum += BiasL2[index2]
            TestLayer2Input[index2] = sum

        TestLayer2Output =  1 / (1 + np.exp(-TestLayer2Input))

        # print("#######################Layer3####################33")
        # Layer3
        L3NodeSum = 0.0
        for index3 in range(L3NodeNum):
            sum = 0.0
            for index2 in range(L2NodeNum):
                sum += TestLayer2Output[index2] * W2[index2][index3]
            sum += BiasL3[index3]
            TestLayer3Input[index3] = sum

        predicted_index = TestLayer3Input.argmax()
        if (label != predicted_index):
            Error_image_count+=1


    #모든 test data에 대해
    print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%\n")
    print("Error Count: ", Error_image_count, "\n")
    print("Correct Rate: ",(TotalTestImg - Error_image_count)/TotalTestImg * 100, "\n")
    epoch += 1








import numpy as np
import sys
import os
from array import array
import random

from struct import *
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#파일 읽기
fp_image = open('C:\\Users\\Administrator\\Documents\\University\\Lab\\MNIST\\training_set\\train-images.idx3-ubyte','rb')
fp_label = open('C:\\Users\\Administrator\\Documents\\University\\Lab\\MNIST\\training_set\\train-labels.idx1-ubyte', 'rb')

#read mnist and show number

#사용할 변수 초기화
img = [] #이미지가 저장될 부분, 28 *28

classifiedImg = [[],[],[],[],[],[],[],[],[],[]] #숫자별로 이미지 저장
L0NodeNum = 784
L1NodeNum = int(input('첫 번째 Hidden Layer Node 갯수를 입력하세요:'))
L2NodeNum = int(input('두 번째 Hidden Layer Node 갯수를 입력하세요:'))
OutputLNodeNum =10
epoch = 0
Error = 1000000000000000
learning_rate = 0.000001
DerivateErrorWithW = 0
count=0
MIN = -1.79769313486e+308
MAX =  100

W0 = np.random.rand(L0NodeNum, L1NodeNum)
W1 = np.random.rand(L1NodeNum, L2NodeNum)
W2 = np.random.rand(L2NodeNum, OutputLNodeNum)


DerivateErrorWithLayer1 = [0] * L1NodeNum
DerivateErrorWithLayer2 = [0] * L2NodeNum
DerivateErrorWithLayer3 = [0] * OutputLNodeNum

BiasL1 = [1] * L1NodeNum
BiasL2 = [1] * L2NodeNum
BiasOutputL = [1] * OutputLNodeNum

HiddenLayer1 = np.zeros(L1NodeNum)
HiddenLayer2 = np.zeros(L2NodeNum)
OutputLayer = np.zeros(OutputLNodeNum)

#참고자료 initialization 활용
"""
for i in range(L0NodeNum):
    for j in range(L1NodeNum):
        W0[i][j] *= np.sqrt(2/L1NodeNum)

for i in range(L1NodeNum):
    for j in range(L2NodeNum):
        W1[i][j] *= np.sqrt(2 / L2NodeNum)

for i in range(L2NodeNum):
    for j in range(OutputLNodeNum):
        W2[i][j] *= np.sqrt(2 / OutputLNodeNum)

"""

s  = fp_image.read(16) #read first 16 byte
ㅣ = fp_label.read(8)  # 1바이트씩 읽음


while True:
    s  = fp_image.read(784) #784 바이트씩 읽음
    ㅣ = fp_label.read(1) #1 바이트 씩 읽음

    if not s:
        break
    if not ㅣ:
        break

    #unpack
    index = int(ㅣ[0])
    img = list(unpack(len(s) * 'B', s)) #byte를 unsigned char 형식으로

#epoch 횟수 지정
while epoch < 1:
    # classifiedImg 마다
    for i in range(10):
        onehot = [0,0,0,0,0,0,0,0,0,0]
        onehot[i] = 1


        for j in range(len(classifiedImg[i])):
            # img 한장한장
            img = classifiedImg[i][j]
            Error = 0

            #print("Layer1")
            #layer1 의 한노드씩
            for index1 in range(L1NodeNum):
                sum = 0.0

                #784개의 모든픽셀
                for index0 in range(L0NodeNum):
                    sum += img[index0] * W0[index0][index1]
                #bias
                sum += BiasL1[index1]

                #ReLU 사용
                HiddenLayer1[index1] = max(sum,0.0)

                #print(sum)


            #print("\n\n\n\n\n\n\n\n\n\n")
            #print("Layer2")
            #layer2의 한노드씩
            for index2 in range(L2NodeNum):
                sum = 0
                #L1의 노드갯수만큼
                for index1 in range(L1NodeNum):
                    sum  = min(MAX, sum +HiddenLayer1[index1] * W1[index1][index2])
                    if sum == MAX:
                        break

                HiddenLayer2[index2] = max(sum,0.0)
                #print(sum)

            for index3 in range(OutputLNodeNum):
                sum = 0
                #OutputLayer 노드 갯수만큼
                for index2 in range(L2NodeNum):
                    sum = min(MAX, sum +HiddenLayer2[index2] * W2[index2][index3])
                    if sum == MAX:
                        break

                OutputLayer[index3] = max(sum,0.0)
                #print(sum)

            for index3 in range(OutputLNodeNum):
                Error += Error + 1/2 *((OutputLayer[index3] - onehot[index3])**2)
                if Error > MAX:
                    Error  = MAX
                    break

            if count % 100 == 0:
                print("Error : ", Error)

            #####################backpropagation###################
            for index3 in range(OutputLNodeNum):
                DerivateErrorWithLayer3[index3] = OutputLayer[index3] - onehot[index3]

            for index2 in range(L2NodeNum):
                derivate = 0
                for index3 in range (OutputLNodeNum):
                    derivate += DerivateErrorWithLayer3[index3] * W2[index2][index3]
                    if derivate>MAX:
                        derivate = MAX
                        break

                DerivateErrorWithLayer2[index2] = derivate

            for index1 in range(L1NodeNum):
                derivate = 0
                for index2 in range(L2NodeNum):
                    derivate += DerivateErrorWithLayer2[index2] * W1[index1][index2]
                    if derivate>MAX:
                        derivate = MAX
                        break

                DerivateErrorWithLayer1[index1] = derivate

            for index0 in range(L0NodeNum):
                for index1 in range(L1NodeNum):
                    W0[index0][index1] = min(MAX, W0[index0][index1] - learning_rate * DerivateErrorWithLayer1[index1] * img[index0])

            for index1 in range(L1NodeNum):
                for index2 in range(L2NodeNum):
                    W1[index1][index2] = min(MAX, W1[index1][index2] - learning_rate * DerivateErrorWithLayer2[index2] * HiddenLayer1[index1])

            for index2 in range(L2NodeNum):
                for index3 in range (OutputLNodeNum):
                    W2[index2][index3] = min(MAX, W2[index2][index3] - learning_rate * DerivateErrorWithLayer3[index3] * HiddenLayer2[index2])

            count+=1

    epoch +=1

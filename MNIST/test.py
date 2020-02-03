import numpy as np

PaddingNum = 0
KernelSize = 3
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

    height, width = img.shape
    matrix = np.zeros(KernelSize * KernelSize)
    for i in range(height - KernelSize + 1):
        for j in range (width - KernelSize + 1):
            array = img[i][j:j+KernelSize]
            for k in np.arange(1, KernelSize):
                array1 = img[i + k][j:j+KernelSize]
                array = np.concatenate((array, array1))
            matrix = np.vstack((matrix, array))
    matrix = np.delete(matrix, 0, axis = 0)
    return matrix

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


# img = np.ones((2, 2, 2))
# cube = np.zeros((2, 4, 4))
# for d in range(2):
#     for h in range(2):
#         for w in range(2):
#             cube[d][h+1][w+1] = img[d][h][w]
# print(cube)

# target = np.reshape(np.arange(27), (3, 3, 3))
# print(target)
# img = np.insert(img, [0, 4], [0, 0], axis=1)
# # img = np.vstack((img, np.zeros(6)))
# img = np.insert(img, 0, np.zeros(6), axis=0)
# img = np.insert(img, 4, np.zeros(4+2), axis=0)
# # img = np.insert(img, [0, 3], [0, 0], axis=0)
# # img[1] = np.append(img[1], 0, axis = 1)

# img1 = np.arange(27)
# img1 = img1.reshape(3, 3, 3)
# print(img1)
# img1 = img1.reshape(1, -1)
# print(img1)
# img1 = img1.reshape(3, 3, 3)
# print(img1)
# print(img1.T)
# print(img1.T[0])
# print(np.array(img1.shape).size)
#
# img1 = ChangeToConvolutionMatrix(img1)
# print(img1)
#
# img1  = np.reshape(img1, (1, 9, 4))
# MaxPoolingL1ResultMatrix = np.zeros((1, 9, 4))
# L1Max = MaxPooling(img1, MaxPoolingL1ResultMatrix)

# img1 = np.arange(4)
# img2 = np.arange(4, 8)
# print(img1)
# print(img2)
# print(img1 * img2)
# print(np.sum(img1 * img2))

# img1 = np.reshape(np.arange(12), (3,4))
# print(img1)
# print(img1.T)
# print(img1)
# # # img2 = img1.reshape(3, 4)
# # print(img2)
# img3 = img1.reshape(3, 2, 2)
# print(img3)
# img1 = img3[:][:][0:1]
# print(img1)

# a = np.ones((3, 3, 3))
# # print(a)
# b = Padding(a, 1)
# # print(b)
# c = b[0:5, 1:4, 1:4]
# print(c)

# d = np.ones((3, 3, 3))
# d[:, 0:1, 0:3] += a[:, 0:1, 0:3]
# print(d)

# f = np.reshape(np.arange(27), (3,3,3))
# print("f", f)
# new = c[:, 0:1, 0:1] + f[:, 0:1, 0:1]
# print(new)

# a  = np.array([1,2,3])
#
# if a>1:
#     a=0
# print(a)


L1 = np.arange(8)
L1 = np.reshape(L1, (2,2,2))
print(L1)


MaxPoolingL1Result = np.zeros((2,2,2))
Filter1Count = 3


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


                        elif index == 1:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h][w + 1] = 1


                        elif index == 2:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h + 1][w] = 1

                        else:
                            if depth == Filter1Count:
                                MaxPoolingL1Result[d][h + 1][w + 1] = 1


                        LayerMax[d][int(h / 2)][int(w / 2)] = value
    return LayerMax

L2 = MaxPooling(L1)
print(L2)
print(L2.shape)

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


MaxPoolingL1Result = np.zeros((2,2,2))
MaxPoolingL1Result[0][0][0] = 1
MaxPoolingL1Result[1][1][1] = 1
MaxPoolingL1Result[0:2][0: 0+2][0 :0+2] *= 10
print(MaxPoolingL1Result)
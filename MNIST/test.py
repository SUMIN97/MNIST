import numpy as np

PaddingNum = 0
def Padding(img, size):
    height, width = img.shape
    for i in range(height):
        for j in range(size): #size 횟수만큼 행의 맨 앞에 num을 추가
            img = np.insert(img[i], 0, PaddingNum)
            #img.insert(img[i], 0, PaddingNum)
        for j in range(size): #size 횟수만큼 행의 맨 뒤에 num을 추가
            np.append(img[i], PaddingNum)
            # img.append(img[i], PaddingNum)
    for j in range(size):
        np.insert(img, 0, np.zeros((width + size + size)))
        np.append(img, np.zeros((width + size + size)))
        # img.insert(img, 0, np.zeros((width + size + size)))
        # img.append(img, np.zeros((width + size + size)))
    return img



img = np.ones((3, 4))
img = np.insert(img, 0, 0, axis=1)
img = np.append(img, 0, axis = 1)
print(img)


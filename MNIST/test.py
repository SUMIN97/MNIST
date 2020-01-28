import numpy as np

img = np.zeros((3, 4))
height, width = img.shape
print(height, width)

array1 = np.array([1,-2,-3])
array2 = np.array([4,5,6])
array3 = np.array([7,8,9])


array1 = np.array([[[1,2,3,4], [5,6, 7, 8]]])
array1.reshape((-1, 2, 2))
print(array1)
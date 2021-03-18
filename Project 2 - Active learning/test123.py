
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


#data = datasets.FashionMNIST(root="Project 2 - Active learning", download=True)
#print(data)

import idx2numpy
import numpy as np
#os.chdir('/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw')
#Getting file paths
file_train = '/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw/train-images-idx3-ubyte'
file_test = '/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw/t10k-images-idx3-ubyte'
file_train_label = '/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw/train-labels-idx1-ubyte'
file_test_label = '/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw/t10k-labels-idx1-ubyte'

#Creating numpy arrays
arr_train = idx2numpy.convert_from_file(file_train)
arr_train_label =  idx2numpy.convert_from_file(file_train_label)
arr_test = idx2numpy.convert_from_file(file_test)
arr_test_label = idx2numpy.convert_from_file(file_test_label)



#cv.imshow("Image", arr[4], )
#plt.imshow(arr_train[4], cmap='gray')
#plt.show()

#Making subsample
#Xtrain = arr_train[:9000]
#ytrain = arr_train_label[:9000]


Xtest = arr_test
ytest = arr_test_label

#Creating pools for pool based active learning
Xtest = Xtest[5000:]
ytest = ytest[5000:]
Xpool = Xtest[:5000]
ypool = ytest[:5000]

print(Xtest.shape, Xpool.shape)











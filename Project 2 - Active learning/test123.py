
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
import sklearn.linear_model as lin
import idx2numpy



#data = datasets.FashionMNIST(root="Project 2 - Active learning", download=True)
#print(data)

#os.chdir('/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw')
#Getting file paths
file_train = '/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw/train-images-idx3-ubyte'
file_test = '/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw/t10k-images-idx3-ubyte'
file_train_label = '/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw/train-labels-idx1-ubyte'
file_test_label = '/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw/t10k-labels-idx1-ubyte'

#Creating numpy arrays
arr_train = idx2numpy.convert_from_file(file_train)
arr_train_label = idx2numpy.convert_from_file(file_train_label)
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
Xtest = np.concatenate(Xtest[5000:]).ravel()
ytest = np.concatenate(ytest[5000:]).ravel()
Xpool = np.concatenate(Xpool[:5000]).ravel()
ypool = np.concatenate(ypool[:5000]).ravel()

#Defining model
lr = lin.LogisticRegression(penalty='l2',C=1.)


addn = 2 #samples to add each time
#randomize order of pool to avoid sampling the same subject sequentially
np.random.seed(0)
order = np.random.permutation(range(len(Xpool)))

#samples in the pool
poolidx=np.arange(len(Xpool),dtype=np.int)
ninit = 10 #initial samples
#initial training set
trainset=order[:ninit]
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
#remove data from pool
poolidx=np.setdiff1d(poolidx,trainset)

model=lr
testacc=[]
for i in range(25):
    #Fit model
    model.fit(Xtrain, ytrain)
    #predict on test set
    ye = model.predict(Xtest)
    #calculate and accuracy and add to list
    testacc.append((ninit+i*addn, sklearn.metrics.accuracy_score(ytest, ye)))
    print('Model: LR, %i random samples'%(ninit+i*addn))















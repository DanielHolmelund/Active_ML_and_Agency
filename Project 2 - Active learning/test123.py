
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
import numpy as np
from PIL import Image
import sklearn
from sklearn.utils import resample
import numpy as np
import os
import matplotlib.pyplot as plt
import sklearn.linear_model as lin
import idx2numpy
from sklearn.ensemble import RandomForestClassifier



#data = datasets.FashionMNIST(root="Project 2 - Active learning", download=True)
#print(data)

os.chdir('/Users/christiandjurhuus/PycharmProjects/Active_ML_and_Agency/Project 2 - Active learning/Project 2 - Active learning/FashionMNIST/raw')
#Getting file paths


file_train = 'train-images-idx3-ubyte'
file_test = 't10k-images-idx3-ubyte'
file_train_label = 'train-labels-idx1-ubyte'
file_test_label = 't10k-labels-idx1-ubyte'

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


X = arr_test
y = arr_test_label

#Creating pools for pool based active learning
Xtest = X[5000:].reshape(5000,784)
ytest = y[5000:]
Xpool = X[:5000].reshape(5000,784)
ypool = y[:5000]


'''
#The following function is from the course "Signals and data"
#Visualizing a subsample of the data
def show_image(x, title="", clim=None, cmap=plt.cm.gray, colorbar=False):
    ax = plt.gca()
    im = ax.imshow(x.reshape((28, 28)), cmap=cmap, clim=clim)

    if len(title) > 0:
        plt.title(title)

    plt.axis('off')

    if colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

num_images_per_row = 10
num_rows = 10
num_images = num_images_per_row * num_rows

plt.figure(figsize=(20, 8))
for i in range(num_images):
    plt.subplot(num_rows, num_images_per_row, 1 + i)
    show_image(Xtest[i, :])
plt.show()
'''


#solver='saga', random_state=0
#Defining model
lr = lin.LogisticRegression(penalty='l2',C=1., max_iter=5000, warm_start=True)
#lr = RandomForestClassifier(40, n_jobs=-1)

#Number of iterations
N = 54

#########################################Randomly increasing the training set##########################################

addn = 10 #samples to add each time
#randomize order of pool to avoid sampling the same subject sequentially
np.random.seed(3)
order = np.random.permutation(range(len(Xpool)))
#samples in the poolx
poolidx=np.arange(len(Xpool),dtype=int)
ninit = 30 #initial samples
#initial training set
trainset=order[:ninit]
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
#remove data from pool
poolidx=np.setdiff1d(poolidx,trainset)

model=lr
testacc=[]

for i in range(N):
    #Fit model
    model.fit(np.take(Xpool, order[:ninit+i*addn], axis = 0), np.take(ypool, order[:ninit+i*addn], axis=0))
    #predict on test set
    y_hat = model.predict(Xtest)
    #calculate and accuracy and add to list
    testacc.append((ninit+i*addn, sklearn.metrics.accuracy_score(ytest, y_hat)))
    print('Model: LR, %i random samples'%(ninit+i*addn))

#Plot learning curve (test set on independent subjects)
'''
plt.plot(*tuple(np.array(testacc).T));
plt.show()
'''

''' 
Så er der frit slag om hvad man vil implementere nedenunder ;-))
'''

###############################################Uncertainty sampling#####################################################

###Least confidence

testacc_al = []
trainset = order[:ninit]
Xtrain = np.take(Xpool, trainset, axis=0)
ytrain = np.take(ypool, trainset, axis=0)
poolidx = np.arange(len(Xpool), dtype=np.int)
poolidx = np.setdiff1d(poolidx, trainset)

for i in range(N):
    # fill out code to select samples according to uncertainty here
    model.fit(Xtrain, ytrain)

    y_hat = model.predict(Xtest)

    testacc_al.append((len(Xtrain), (y_hat == ytest).mean()))

    # Getting label probabilities
    p = model.predict_proba(Xpool[poolidx])
    # Sorting the probabilites to find the least confident
    p_sort_idx = np.argsort(1 - p.max(1))

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[p_sort_idx[-addn:]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[p_sort_idx[-addn:]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[p_sort_idx[-addn:]])

    print('Model: LR, %i Least confident sampling' % (len(Xtrain)))


'''#Plot learning curve
plt.plot(*tuple(np.array(testacc).T))
plt.plot(*tuple(np.array(testacc_al).T))
plt.legend(('random sampling','uncertainty sampling'))
plt.show()
'''


#Margin sampling

testacc_al_LM = []
trainset = order[:ninit]
Xtrain = np.take(Xpool, trainset, axis=0)
ytrain = np.take(ypool, trainset, axis=0)
poolidx = np.arange(len(Xpool), dtype=np.int)
poolidx = np.setdiff1d(poolidx, trainset)

for i in range(N):
    # fill out code to select samples according to uncertainty here
    model.fit(Xtrain, ytrain)

    y_hat = model.predict(Xtest)

    testacc_al_LM.append((len(Xtrain), (y_hat == ytest).mean()))

    # Getting label probabilities
    p = model.predict_proba(Xpool[poolidx])
    # Sorting the probabilites to find the least confident

    ix = np.arange(len(p))
    p2, p1 = p.argsort(1)[:, -2:].T
    res = p[ix, p1] - p[ix, p2]
    p_LM_sort = np.argsort(res)

#    p_LM_sort = np.argsort(np.sort(p, axis=1)[:,-1] - np.sort(p, axis=1)[:,-2] )

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[p_LM_sort[:addn]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[p_LM_sort[:addn]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[p_LM_sort[:addn]])

    print('Model: LR, %i Maximum margin sampling' % (len(Xtrain)))

#Entropy

testacc_al_Entropy = []
trainset = order[:ninit]
Xtrain = np.take(Xpool, trainset, axis=0)
ytrain = np.take(ypool, trainset, axis=0)
poolidx = np.arange(len(Xpool), dtype=np.int)
poolidx = np.setdiff1d(poolidx, trainset)

for i in range(N):
    # fill out code to select samples according to uncertainty here
    model.fit(Xtrain, ytrain)

    y_hat = model.predict(Xtest)

    testacc_al_Entropy.append((len(Xtrain), (y_hat == ytest).mean()))

    # Getting label probabilities
    p = model.predict_proba(Xpool[poolidx])
    # Sorting the probabilites to find the least confident
    res = -np.sum(p*np.log2(p+1e-8),axis=1) #adding small value to avoid 0 in logarithm
    idx_entropy = res.argsort()

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[idx_entropy[-addn:]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[idx_entropy[-addn:]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[idx_entropy[-addn:]])

    print('Model: LR, %i Entropy samples' % (len(Xtrain)))

#Determine 95% confidence intervals
#RM
x_samples = [x[0] for x in testacc]
x_samples = np.asarray(x_samples)
accuracies_RS = [x[1] for x in testacc]
accuracies_RS = np.asarray(accuracies_RS)
CI_RS = 1.96 * np.sqrt((accuracies_RS*(1-accuracies_RS))/x_samples)
#LS
accuracies_LS = [x[1] for x in testacc_al]
accuracies_LS = np.asarray(accuracies_LS)
CI_AL = 1.96 * np.sqrt((accuracies_LS*(1-accuracies_LS))/x_samples)
#MS
accuracies_MS = [x[1] for x in testacc_al_LM]
accuracies_MS = np.asarray(accuracies_MS)
CI_MS = 1.96 * np.sqrt((accuracies_MS*(1-accuracies_MS))/x_samples)
#E
accuracies_E = [x[1] for x in testacc_al_Entropy]
accuracies_E = np.asarray(accuracies_E)
CI_E = 1.96 * np.sqrt((accuracies_E*(1-accuracies_E))/x_samples)

#Uncomment the errorbar commands to display 95% confidence interval

#Plot learning curve
plt.plot(*tuple(np.array(testacc_al).T))
plt.errorbar(*tuple(np.array(testacc_al).T), CI_AL, color= 'orange', elinewidth=None)

plt.plot(*tuple(np.array(testacc_al_LM).T))
plt.errorbar(*tuple(np.array(testacc_al_LM).T), CI_MS, color= 'green', elinewidth=None)

plt.plot(*tuple(np.array(testacc_al_Entropy).T))
plt.errorbar(*tuple(np.array(testacc_al_Entropy).T), CI_E, color='red', elinewidth=None)

plt.plot(*tuple(np.array(testacc).T))
plt.errorbar(*tuple(np.array(testacc).T), CI_RS, color= 'blue', elinewidth=None)

plt.grid()
plt.legend(('random sampling','LC','MS','Entropy'))
plt.xlabel('Samples', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel('Samples', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.savefig('Uncertainty_sampling_final.png')
plt.show()


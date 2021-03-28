
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



#Defining model
lr = lin.LogisticRegression(penalty='l2',C=1., max_iter=5000)


#########################################Randomly increasing the training set##########################################

addn = 1 #samples to add each time
#randomize order of pool to avoid sampling the same subject sequentially
np.random.seed(0)
order = np.random.permutation(range(len(Xpool)))
#samples in the poolx
poolidx=np.arange(len(Xpool),dtype=int)
ninit = 20 #initial samples
#initial training set
trainset=order[:ninit]
print(trainset)
Xtrain=np.take(Xpool,trainset,axis=0)
ytrain=np.take(ypool,trainset,axis=0)
#remove data from pool
poolidx=np.setdiff1d(poolidx,trainset)

model=lr
testacc=[]

for i in range(25):
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

for i in range(25):
    # fill out code to select samples according to uncertainty here
    model.fit(Xtrain, ytrain)

    y_hat = model.predict(Xtest)

    testacc_al.append((len(Xtrain), (y_hat == ytest).mean()))

    # Getting label probabilities
    p = model.predict_proba(Xpool[poolidx])
    # Sorting the probabilites to find the least confident
    p_sort = np.argsort(1 - p.max(1))

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[p_sort[-addn:]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[p_sort[-addn:]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[p_sort[-addn:]])

    print('Model: LR, %i random samples' % (len(Xtrain)))


'''#Plot learning curve
plt.plot(*tuple(np.array(testacc).T))
plt.plot(*tuple(np.array(testacc_al).T))
plt.legend(('random sampling','uncertainty sampling'))
plt.show()
'''


#Largest margin

testacc_al_LM = []
trainset = order[:ninit]
Xtrain = np.take(Xpool, trainset, axis=0)
ytrain = np.take(ypool, trainset, axis=0)
poolidx = np.arange(len(Xpool), dtype=np.int)
poolidx = np.setdiff1d(poolidx, trainset)

for i in range(25):
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
    p_LM_sort = np.argsort(1 - res)

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[p_LM_sort[-addn:]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[p_LM_sort[-addn:]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[p_LM_sort[addn:]])

    print('Model: LR, %i random samples' % (len(Xtrain)))

#Entropy

testacc_al_Entropy = []
trainset = order[:ninit]
Xtrain = np.take(Xpool, trainset, axis=0)
ytrain = np.take(ypool, trainset, axis=0)
poolidx = np.arange(len(Xpool), dtype=np.int)
poolidx = np.setdiff1d(poolidx, trainset)

for i in range(25):
    # fill out code to select samples according to uncertainty here
    model.fit(Xtrain, ytrain)

    y_hat = model.predict(Xtest)

    testacc_al_Entropy.append((len(Xtrain), (y_hat == ytest).mean()))

    # Getting label probabilities
    p = model.predict_proba(Xpool[poolidx])
    # Sorting the probabilites to find the least confident
    res = -np.sum(p*np.log2(p),1)
    idx_entropy = res.argsort()

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[idx_entropy[-addn:]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[idx_entropy[-addn:]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[idx_entropy[-addn:]])

    print('Model: LR, %i random samples' % (len(Xtrain)))



#Plot learning curve
plt.plot(*tuple(np.array(testacc).T))
plt.plot(*tuple(np.array(testacc_al).T))
plt.plot(*tuple(np.array(testacc_al_LM).T))
plt.plot(*tuple(np.array(testacc_al_Entropy).T))
plt.legend(('random sampling','LC','LM','Entropy'))
plt.show()

'''
###############################################Query by commitee########################################################
testacc_qbc_LC = []
ncomm = 10
trainset = order[:ninit]
Xtrain = np.take(Xpool, trainset, axis=0)
ytrain = np.take(ypool, trainset, axis=0)
poolidx = np.arange(len(Xpool), dtype=np.int)
poolidx = np.setdiff1d(poolidx, trainset)
for i in range(25):
    # fill out code to do QBC by bootstrapping a commitee of LR models
#    labels = np.zeros((ncomm, len(Xpool[poolidx]), 10))
    labels = []
    labels_entropy = np.zeros((ncomm,len(Xpool[poolidx]),10))
    model.fit(Xtrain, ytrain)

    y_hat = model.predict(Xtest)

    testacc_qbc_LC.append((len(Xtrain), (y_hat == ytest).mean()))

    for j in range(ncomm):
        xtr, ytr = resample(Xtrain, ytrain, n_samples=len(Xtrain), replace=True, stratify=ytrain)
        model.fit(xtr, ytr)
        labels.append(model.predict(Xpool[poolidx]))

    #Least confident

    ypool_p=(np.mean(np.array(labels)==0,0),
             np.mean(np.array(labels)==1,0),
             np.mean(np.array(labels)==2,0),
             np.mean(np.array(labels)==3,0),
             np.mean(np.array(labels)==4,0),
             np.mean(np.array(labels)==5,0),
             np.mean(np.array(labels)==6,0),
             np.mean(np.array(labels)==7,0),
             np.mean(np.array(labels)==8,0),
             np.mean(np.array(labels)==9,0))
    ypool_p = np.array(ypool_p).T
    ypool_p_idx = np.argsort(-np.max(ypool_p, 1)) #Least confident

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[ypool_p_idx[-addn:]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[ypool_p_idx[-addn:]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[ypool_p_idx[-addn:]])

    print('Model: LR, %i random samples' % (len(Xtrain)))

#Plot learning curve
plt.plot(*tuple(np.array(testacc).T));
plt.plot(*tuple(np.array(testacc_al).T));
plt.plot(*tuple(np.array(testacc_qbc_LC).T));
plt.legend(('random sampling', 'Uncertainty sampling','QBC LC'));
plt.show()

#Trying to implement standard entropy
testacc_qbc_entropy = []
ncomm = 10
trainset = order[:ninit]
Xtrain = np.take(Xpool, trainset, axis=0)
ytrain = np.take(ypool, trainset, axis=0)
poolidx = np.arange(len(Xpool), dtype=np.int)
poolidx = np.setdiff1d(poolidx, trainset)
for i in range(25):
    # fill out code to do QBC by bootstrapping a commitee of LR models
#    labels = np.zeros((ncomm, len(Xpool[poolidx]), 10))
    labels_entropy = np.zeros((ncomm,len(Xpool[poolidx]),10))
    model.fit(Xtrain, ytrain)

    y_hat = model.predict(Xtest)

    testacc_qbc_entropy.append((len(Xtrain), (y_hat == ytest).mean()))

    for j in range(ncomm):
        xtr, ytr = resample(Xtrain, ytrain, n_samples=len(Xtrain), replace=True, stratify=ytrain)
        model.fit(xtr, ytr)
        labels_entropy[j] = model.predict_proba(Xpool[poolidx])

    #Entropy
    res = -np.sum(labels_entropy * np.log2(labels_entropy), 1)
    idx = np.argmax(res)

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[idx[-addn:]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[idx[-addn:]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[idx[-addn:]])

    print('Model: LR, %i random samples' % (len(Xtrain)))

#Plot learning curve
plt.plot(*tuple(np.array(testacc).T));
plt.plot(*tuple(np.array(testacc_al).T));
plt.plot(*tuple(np.array(testacc_qbc_LC).T));
plt.plot(*tuple(np.array(testacc_qbc_entropy).T));
plt.legend(('random sampling', 'Uncertainty sampling','QBC LC', 'QBC Vote entropy'));
plt.show()
'''
'''
#Vote Entropy
testacc_qbc_vote_entropy = []
ncomm = 10
trainset = order[:ninit]
Xtrain = np.take(Xpool, trainset, axis=0)
ytrain = np.take(ypool, trainset, axis=0)
poolidx = np.arange(len(Xpool), dtype=np.int)
poolidx = np.setdiff1d(poolidx, trainset)
for i in range(25):
    # fill out code to do QBC by bootstrapping a commitee of LR models
#    labels = np.zeros((ncomm, len(Xpool[poolidx]), 10))
    labels_entropy = np.zeros((ncomm,len(Xpool[poolidx]),10))
    model.fit(Xtrain, ytrain)

    y_hat = model.predict(Xtest)

    testacc_qbc_vote_entropy.append((len(Xtrain), (y_hat == ytest).mean()))

    for j in range(ncomm):
        xtr, ytr = resample(Xtrain, ytrain, n_samples=len(Xtrain), replace=True, stratify=ytrain)
        model.fit(xtr, ytr)
        labels_entropy[j] = model.predict_proba(Xpool[poolidx])

    #Vote Entropy
    disagree = -np.abs(0.5 - (labels_entropy[:,:,0] > 0.5).mean(0))  # max disagreement
    disagree_sort = np.argsort(disagree)

    # Now lets add them to the training set and remove them from the pool
    # adding
    Xtrain = np.concatenate((Xtrain, Xpool[poolidx[disagree_sort[-addn:]]]))
    ytrain = np.concatenate((ytrain, ypool[poolidx[disagree_sort[-addn:]]]))

    # removing
    poolidx = np.setdiff1d(poolidx, poolidx[disagree_sort[-addn:]])

    print('Model: LR, %i random samples' % (len(Xtrain)))

#Plot learning curve
plt.plot(*tuple(np.array(testacc).T));
plt.plot(*tuple(np.array(testacc_al).T));
plt.plot(*tuple(np.array(testacc_qbc_LC).T));
plt.plot(*tuple(np.array(testacc_qbc_entropy).T));
plt.legend(('random sampling', 'Uncertainty sampling','QBC LC', 'QBC Vote entropy'));
plt.show()
'''


###############################################Expected Improvement####################################################




###############################################Density weighting#######################################################


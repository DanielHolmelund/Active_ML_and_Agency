#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:40:51 2021

@author: christian djurhuus, Daniel Holmelund & Felix Burmester
"""


import os 
import numpy as np
import pandas as pd
import csv
import scipy.stats
from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
from scipy.io import loadmat
from scipy import stats
from toolbox_02450 import dbplot, dbprobplot, mcnemar
import sklearn.tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import shuffle
from sklearn import model_selection
from sklearn.preprocessing import LabelEncoder
import statsmodels.api as sm
from statsmodels.formula.api import ols
import statsmodels.api as sm 
import pylab as py 

np.random.seed(0)

os.chdir("/Users/christiandjurhuus/Desktop/3_ugers")

filename = "wasall_02445.txt"

df = pd.read_csv("wasall_02445.txt", delimiter="\t")
df_diag = pd.read_csv("wasall_02445_fixed.txt", delimiter="\t")

df.info()

#Shuffle dataset to eliminate possible trends.
df = sklearn.utils.shuffle(df)
df_diag = sklearn.utils.shuffle(df_diag)

del df["obsnr"]
del df_diag["obsnr"]

'''
#Normalizing symmetry measures of original data
df["S"] = (df["S"]-df["S"].mean())/df["S"].std()
df["W"] = (df["W"]-df["W"].mean())/df["W"].std()
df["A"] = (df["A"]-df["A"].mean())/df["A"].std()

#Normalizing symmetry measures of diagonal data
df_diag["S"] = (df_diag["S"]-df_diag["S"].mean())/df_diag["S"].std()
df_diag["W"] = (df_diag["W"]-df_diag["W"].mean())/df_diag["W"].std()
df_diag["A"] = (df_diag["A"]-df_diag["A"].mean())/df_diag["A"].std()
'''

#Making the three different data sets

#A/W

df_AW = df[["A","W"]]

#PC3/PC4

df_PC = df[["PC3","PC4"]]

#Combined

df_com = df[["A","W","PC3","PC4"]]


#Altered data set
#A/W

df_diag_AW = df_diag[["A","W"]]

#PC3/PC4

df_diag_PC = df_diag[["PC3","PC4"]]

#Combined

df_diag_com = df_diag[["A","W","PC3","PC4"]]




#Determining number of trees
'''
y = df["engTreat5"].to_numpy()
N = len(y)

x_AW = df_AW.to_numpy()

CV = model_selection.LeaveOneOut()
L = 10
errors = np.zeros((N,L))
i=0


#Trying to determine a suitable number of trees in the random forest
for train_index, test_index in CV.split(x_AW, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = x_AW[train_index,:]
    y_train = y[train_index]
    X_test = x_AW[test_index,:]
    y_test = y[test_index]
    
    
    #Fit classifier and vary the amount of baggings
    for l in range(1, L + 1 ):
        # Fit model using random tree classifier:
        rf_classifier_AW = RandomForestClassifier(l*10,criterion = "gini")
        
        rf_classifier_AW.fit(X_train, y_train)
        y_est_train = rf_classifier_AW.predict(X_train).T
        y_est_test = rf_classifier_AW.predict(X_test).T
        
        TrainErrorRate = (y_train!=y_est_train).sum(dtype=float)/N
        TestErrorRate = (y_test!=y_est_test).sum(dtype=float)/N
        errors[i,l-1] = TestErrorRate
        
#        print('Train error rate A/W: {:.2f}%'.format(ErrorRate*100))
        
#        print(pd.Series(rf_classifier_AW.feature_importances_, 
#                        index=df_AW.columns).sort_values(ascending=False))  
    i += 1



figure()
plot(100*sum(errors,0)/N)
xlabel('Number of baggings')
ylabel('Classification error rate (%)')
show()
'''


#Determining the training and test error using LOO

y = df["engTreat5"].to_numpy()
N = len(y)

x_AW = df_AW.to_numpy()
x_PC = df_PC.to_numpy()
x_com = df_com.to_numpy()

CV = model_selection.LeaveOneOut()
i=0


error_train_AW = np.zeros(N)
error_test_AW = np.zeros(N)

error_train_PC = np.zeros(N)
error_test_PC = np.zeros(N)

error_train_com = np.zeros(N)
error_test_com = np.zeros(N)

AW_est_test = np.array([])

np.random.seed(0)

for train_index, test_index in CV.split(x_AW, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = x_AW[train_index,:]
    y_train = y[train_index]
    X_test = x_AW[test_index,:]
    y_test = y[test_index]
    L = 160


# Fit model using random tree classifier:
    rf_classifier_AW = RandomForestClassifier(L,criterion = "gini", random_state = 42, min_samples_leaf = 7, max_features="auto", n_jobs=-1)
    
    rf_classifier_AW.fit(X_train, y_train)
    y_est_train = rf_classifier_AW.predict(X_train).T
    y_est_test = rf_classifier_AW.predict(X_test).T
    AW_est_test = np.append(AW_est_test, y_est_test)
    
    TrainErrorRate = (y_train!=y_est_train).sum(dtype=float)/float(len(y_est_train))
    TestErrorRate = (y_test!=y_est_test).sum(dtype=float)/float(len(y_est_test))

    error_train_AW[i] = TrainErrorRate
    error_test_AW[i] = TestErrorRate

    i += 1
    
print('Mean train error rate A/W: {:.2f}%'.format(np.mean(100*(error_train_AW))))
print('Mean test error rate A/W: {:.2f}%'.format(np.mean(100*(error_test_AW))))

i=0
PC_est_test = np.array([])
np.random.seed(0)
for train_index, test_index in CV.split(x_PC, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = x_PC[train_index,:]
    y_train = y[train_index]
    X_test = x_PC[test_index,:]
    y_test = y[test_index]
    L = 160
    
#Fit classifier

# Fit model using random tree classifier:
    rf_classifier_PC = RandomForestClassifier(L,criterion = "gini", random_state = 42,min_samples_leaf = 7, max_features="auto", n_jobs=-1)
    
    rf_classifier_PC.fit(X_train, y_train)
    y_est_train_PC = rf_classifier_PC.predict(X_train).T
    y_est_test_PC = rf_classifier_PC.predict(X_test).T
    PC_est_test = np.append(PC_est_test, y_est_test_PC)
    
    TrainErrorRate_PC = (y_train!=y_est_train_PC).sum(dtype=float)/float(len(y_est_train_PC))
    TestErrorRate_PC = (y_test!=y_est_test_PC).sum(dtype=float)/float(len(y_est_test_PC))

    error_train_PC[i] = TrainErrorRate_PC
    error_test_PC[i] = TestErrorRate_PC

    i += 1
    
print('Mean train error rate PC: {:.2f}%'.format(np.mean(100*error_train_PC)))
print('Mean test error rate PC: {:.2f}%'.format(np.mean(100*error_test_PC)))

i = 0
com_est_test = np.array([])
np.random.seed(0)
for train_index, test_index in CV.split(x_com, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = x_com[train_index,:]
    y_train = y[train_index]
    X_test = x_com[test_index,:]
    y_test = y[test_index]
    L = 160
    
#Fit classifier

# Fit model using random tree classifier:
    rf_classifier_com = RandomForestClassifier(L,criterion = "gini", random_state = 42, min_samples_leaf = 7, max_features="auto", n_jobs=-1)
    
    rf_classifier_com.fit(X_train, y_train)
    y_est_train_com = rf_classifier_com.predict(X_train).T
    y_est_test_com = rf_classifier_com.predict(X_test).T
    com_est_test = np.append(com_est_test,y_est_test_com)
    
    TrainErrorRate_com = (y_train!=y_est_train_com).sum(dtype=float)/float(len(y_est_train_com))
    TestErrorRate_com = (y_test!=y_est_test_com).sum(dtype=float)/float(len(y_est_test_com))

    error_train_com[i] = TrainErrorRate_com
    error_test_com[i] = TestErrorRate_com

    i += 1
    
print('Mean train error rate com: {:.2f}%'.format(np.mean(100*error_train_com)))
print('Mean test error rate com: {:.2f}%'.format(np.mean(100*error_test_com)))

# True
y_true = np.asarray(df[["engTreat5"]])

# Normalize labels
le = LabelEncoder()
le.fit(["Normal", "Left-fore", "Right-fore", "Left-hind", "Right-hind"])
list(le.classes_)

### Pairwise Mcnemar tests
# McNemar between AW and PC
mcnemar(le.transform(y_true),le.transform(AW_est_test), le.transform(PC_est_test))
# AW and com
mcnemar(le.transform(y_true),le.transform(AW_est_test), le.transform(com_est_test))
# PC and com
mcnemar(le.transform(y_true),le.transform(PC_est_test), le.transform(com_est_test))


##################Altered data ###############################################

y_diag = df_diag["engTreat5"].to_numpy()
N_diag = len(y_diag)

x_diag_AW = df_diag_AW.to_numpy()
x_diag_PC = df_diag_PC.to_numpy()
x_diag_com = df_diag_com.to_numpy()

CV = model_selection.LeaveOneOut()
i=0


error_train_diag_AW = np.zeros(N)
error_test_diag_AW = np.zeros(N)

error_train_diag_PC = np.zeros(N)
error_test_diag_PC = np.zeros(N)

error_train_diag_com = np.zeros(N)
error_test_diag_com = np.zeros(N)

AW_diag_est_test = np.array([])

np.random.seed(0)

for train_index, test_index in CV.split(x_diag_AW, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = x_diag_AW[train_index,:]
    y_train = y_diag[train_index]
    X_test = x_diag_AW[test_index,:]
    y_test = y_diag[test_index]
    L = 160


# Fit model using random tree classifier:
    rf_classifier_AW_diag = RandomForestClassifier(L,criterion = "gini", random_state = 42, min_samples_leaf = 7, max_features="auto", n_jobs=-1)
    
    rf_classifier_AW_diag.fit(X_train, y_train)
    y_est_train_diag_AW = rf_classifier_AW_diag.predict(X_train).T
    y_est_test_diag_AW = rf_classifier_AW_diag.predict(X_test).T
    AW_diag_est_test = np.append(AW_diag_est_test, y_est_test_diag_AW)
    
    TrainErrorRate_diag_AW = (y_train!=y_est_train_diag_AW).sum(dtype=float)/float(len(y_est_train_diag_AW))
    TestErrorRate_diag_AW = (y_test!=y_est_test_diag_AW).sum(dtype=float)/float(len(y_est_test_diag_AW))

    error_train_diag_AW[i] = TrainErrorRate_diag_AW
    error_test_diag_AW[i] = TestErrorRate_diag_AW

    i += 1
    
print('Mean train error rate A/W: {:.2f}%'.format(np.mean(100*(error_train_diag_AW))))
print('Mean test error rate A/W: {:.2f}%'.format(np.mean(100*(error_test_diag_AW))))


i=0
PC_diag_est_test = np.array([])
np.random.seed(0)
for train_index, test_index in CV.split(x_PC, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = x_diag_PC[train_index,:]
    y_train = y_diag[train_index]
    X_test = x_diag_PC[test_index,:]
    y_test = y_diag[test_index]
    L = 160
    


# Fit model using random tree classifier:
    rf_classifier_PC_diag = RandomForestClassifier(L,criterion = "gini", random_state = 42,min_samples_leaf = 7, max_features="auto", n_jobs=-1)
    
    rf_classifier_PC_diag.fit(X_train, y_train)
    y_est_train_diag_PC = rf_classifier_PC_diag.predict(X_train).T
    y_est_test_diag_PC = rf_classifier_PC_diag.predict(X_test).T
    PC_diag_est_test = np.append(PC_diag_est_test, y_est_test_diag_PC)
    
    TrainErrorRate_diag_PC = (y_train!=y_est_train_diag_PC).sum(dtype=float)/float(len(y_est_train_diag_PC))
    TestErrorRate_diag_PC = (y_test!=y_est_test_diag_PC).sum(dtype=float)/float(len(y_est_test_diag_PC))

    error_train_diag_PC[i] =  TrainErrorRate_diag_PC
    error_test_diag_PC[i] = TestErrorRate_diag_PC

    i += 1
    
print('Mean train error rate PC: {:.2f}%'.format(np.mean(100*error_train_diag_PC)))
print('Mean test error rate PC: {:.2f}%'.format(np.mean(100*error_test_diag_PC)))

i = 0
com_diag_est_test = np.array([])
np.random.seed(0)
for train_index, test_index in CV.split(x_com, y):
    print('Crossvalidation fold: {0}/{1}'.format(i+1,N))    
    
    # extract training and test set for current CV fold
    X_train = x_diag_com[train_index,:]
    y_train = y_diag[train_index]
    X_test = x_diag_com[test_index,:]
    y_test = y_diag[test_index]
    L = 160
    

# Fit model using random tree classifier:
    rf_classifier_com_diag = RandomForestClassifier(L,criterion = "gini", random_state = 42, min_samples_leaf = 7, max_features="auto", n_jobs=-1)
    
    rf_classifier_com_diag.fit(X_train, y_train)
    y_est_train_diag_com = rf_classifier_com_diag.predict(X_train).T
    y_est_test_diag_com = rf_classifier_com_diag.predict(X_test).T
    com_diag_est_test = np.append(com_diag_est_test,y_est_test_diag_com)
    
    TrainErrorRate_diag_com = (y_train!=y_est_train_diag_com).sum(dtype=float)/float(len(y_est_train_diag_com))
    TestErrorRate_diag_com = (y_test!=y_est_test_diag_com).sum(dtype=float)/float(len(y_est_test_diag_com))

    error_train_diag_com[i] = TrainErrorRate_diag_com
    error_test_diag_com[i] = TestErrorRate_diag_com

    i += 1
    
print('Mean train error rate com: {:.2f}%'.format(np.mean(100*error_train_diag_com)))
print('Mean test error rate com: {:.2f}%'.format(np.mean(100*error_test_diag_com)))

# True
y_true_diag = np.asarray(df_diag[["engTreat5"]])

# Normalize labels
le2 = LabelEncoder()
le2.fit(["Normal", "Right-fore_Left-hind", "Left-fore_Right-hind"])
list(le.classes_)

### Pairwise Mcnemar tests
# McNemar between AW and PC
mcnemar(le2.transform(y_true_diag),le2.transform(AW_diag_est_test), le2.transform(PC_diag_est_test))
# AW and com
mcnemar(le2.transform(y_true_diag),le2.transform(AW_diag_est_test), le2.transform(com_diag_est_test))
# PC and com
mcnemar(le2.transform(y_true_diag),le2.transform(PC_diag_est_test), le2.transform(com_diag_est_test))



#######Trying to make altered version of the McNemar script from ML & Datamining
# perform McNemars test
def altered_McNemar(y_true_1, y_true_2, yhatA, yhatB, alpha=0.05):
    nn = np.zeros((2,2))
    c1 = yhatA - y_true_1 == 0
    c2 = yhatB - y_true_2 == 0

    nn[0,0] = sum(c1 & c2)
    nn[0,1] = sum(c1 & ~c2)
    nn[1,0] = sum(~c1 & c2)
    nn[1,1] = sum(~c1 & ~c2)

    n = sum(nn.flat);
    n12 = nn[0,1]
    n21 = nn[1,0]

    thetahat = (n12-n21)/n
    Etheta = thetahat

    Q = n**2 * (n+1) * (Etheta+1) * (1-Etheta) / ( (n*(n12+n21) - (n12-n21)**2) )

    p = (Etheta + 1)*0.5 * (Q-1)
    q = (1-Etheta)*0.5 * (Q-1)

    CI = tuple(lm * 2 - 1 for lm in scipy.stats.beta.interval(1-alpha, a=p, b=q) )

    p = 2*scipy.stats.binom.cdf(min([n12,n21]), n=n12+n21, p=0.5)
    print("Result of McNemars test using alpha=", alpha)
    print("Comparison matrix n")
    print(nn)
    if n12+n21 <= 10:
        print("Warning, n12+n21 is low: n12+n21=",(n12+n21))

    print("theta_hat:",thetahat)
    print("Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] = ", CI)
    print("p-value for two-sided test A and B have same accuracy (exact binomial test): p=", p)

    
#Trying to make comparison between original data and altered data.

altered_McNemar(le.transform(y_true), le2.transform(y_true_diag), le.transform(AW_est_test), le2.transform(AW_diag_est_test))
altered_McNemar(le.transform(y_true), le2.transform(y_true_diag), le.transform(PC_est_test), le2.transform(PC_diag_est_test))
altered_McNemar(le.transform(y_true), le2.transform(y_true_diag), le.transform(com_est_test), le2.transform(com_diag_est_test))








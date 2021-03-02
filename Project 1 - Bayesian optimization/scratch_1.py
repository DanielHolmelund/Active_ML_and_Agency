import numpy as np
import matplotlib.pyplot as plt
import GPyOpt
import pandas as pd
import os
import time
from torchvision import datasets, transforms, utils
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterSampler, RandomizedSearchCV, cross_val_score, train_test_split
from scipy.stats import uniform
from sklearn.utils import shuffle
import random
filename = "wasall_02445.txt"
df = pd.read_csv("wasall_02445.txt", delimiter="\t")
del df["obsnr"]

#Making the three different data sets (Inspirred by previous assignment in other course)

Xtrain, Xtest, ytrain, ytest = train_test_split(df[["A","W"]], df[["engTreat5"]], train_size=0.7, test_size=0.3)
model = RandomForestClassifier(n_estimators=int(param[0]), max_depth=int(param[1]), max_features=max_f,
                                       criterion=crit, min_impurity_decrease= float(param[4]), ccp_alpha = float(param[5]), oob_score=True, n_jobs=-1)

# fit the model
model.fit(Xtrain, ytrain)
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
#Setting seed
np.random.seed(34) #Seed 34 works
random.seed(34)

####################Download data and prepping data####################
filename = "wasall_02445_fixed.txt"
df = pd.read_csv("wasall_02445_fixed.txt", delimiter="\t")
del df["obsnr"]

#Shuffle dataset to eliminate possible trends.
df = shuffle(df)


#Making the three different data sets (Inspirred by previous assignment in other course)


#Making train and test set.
#We will start with df_AW

#Temporary hold out method.

def split_by_horse(df, horse):
    test = df[df['horse'].str.match(horse)]
    train = df.drop(df[df['horse'].str.match(horse)].index)
    test = test.reset_index(drop=True)
    train = train.reset_index(drop=True)
    return test, train



Xtrain, Xtest, ytrain, ytest = train_test_split(df[["A","W"]], df[["engTreat5"]], train_size=0.7, test_size=0.3)
horses = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9"]
iter = 25
xs_matrix = np.empty((len(horses), iter))
y_bo_matrix = np.empty((len(horses), iter))
j = 0
param_list = ["A", "W"]
for horse in horses:
    Xtest, Xtrain = split_by_horse(df, horse)
    Xtest = Xtest[["A", "W"]]
    Xtrain = Xtrain[["A", "W"]]
    ytest, ytrain = split_by_horse(df, horse)
    ytest = ytest[["engTreat5"]]
    ytrain = ytrain[["engTreat5"]]
###########################Random search###############################

    # hyperparams dictionary
    domain = {"n_estimators": range(1, 101),
              "criterion": ['gini', 'entropy'],
              "max_depth": range(10, 60, 5),
              "max_features": ['sqrt', 'log2'],
              "min_impurity_decrease": uniform(0, 0.2),
              "ccp_alpha": uniform(0, 0.03)}
    # np.linspace(0, 0.05, 50, dtype=np.float
    # rs = RandomizedSearchCV(model, param_distributions=domain, cv=3, verbose =2, n_iter=10)
    # rs.fit(Xtrain, ytrain)

    # create the ParameterSampler
    param_list = list(ParameterSampler(domain, n_iter=iter, random_state=32))
    print('Param list')
    print(param_list)
    # rounded_list = [dict((k,v) for (k, v) in d.items()) for d in param_list]

    # print('Random parameters we are going to consider')
    # print(rounded_list)

    ## now we can train the random forest using these parameters tuple, and for
    ## each iteration we store the best value of the oob

    current_best_oob = 0
    iteration_best_oob = 0
    max_oob_per_iteration = []
    i = 0
    for params in param_list:
        print(i)
        print(params)
        #    model = RandomForestClassifier(**params, oob_score=True)
        model = RandomForestClassifier(n_estimators=params['n_estimators'], criterion=params['criterion'],
                                       max_depth=params['max_depth'], max_features=params['max_features'],
                                       min_impurity_decrease=params["min_impurity_decrease"], ccp_alpha=params["ccp_alpha"],
                                       oob_score=True)
        start = time.time()
        model.fit(Xtrain, ytrain)
        end = time.time()
        model_oob = model.oob_score_
        print('OOB found:', model_oob)
        if model_oob > current_best_oob:
            current_best_oob = model_oob
            iteration_best_oob = i

        max_oob_per_iteration.append(1 - current_best_oob)
        i += 1
        print(f'It took {end - start} seconds')

    ########################Bayesian Optimization##########################
    # define the domain of the considered parameters
    n_estimators = tuple(np.arange(1, 101, 1, dtype=np.int))
    # print(n_estimators)
    max_depth = tuple(np.arange(10, 110, 10, dtype=np.int))
    # max_features = ('log2', 'sqrt', None)
    max_features = (0, 1)
    # criterion = ('gini', 'entropy')
    criterion = (0, 1)

    min_impurity_decrease = tuple(np.linspace(0, 0.05, 50, dtype=np.float))
    ccp_alpha = tuple(np.linspace(0, 0.03, 60, dtype= np.float))

    # define the dictionary for GPyOpt
    domain = [{'name': 'n_estimators', 'type': 'discrete', 'domain': n_estimators},
              {'name': 'max_depth', 'type': 'discrete', 'domain': max_depth},
              {'name': 'max_features', 'type': 'categorical', 'domain': max_features},
              {'name': 'criterion', 'type': 'categorical', 'domain': criterion},
              {"name": "min_impurity_decrease", "type": "continuous", "domain": (0, 0.05)},
              {"name": "ccp_alpha", "type": "continuous", "domain": (0, 0.03)}]


    ## we have to define the function we want to maximize --> validation accuracy,
    ## note it should take a 2D ndarray but it is ok that it assumes only one point
    ## in this setting
    def objective_function(x):
        # print(x)
        # we have to handle the categorical variables that is convert 0/1 to labels
        # log2/sqrt and gini/entropy
        param = x[0]
        print(param)
        # we have to handle the categorical variables
        if param[2] == 0:
            max_f = 'log2'
        elif param[2] == 1:
            max_f = 'sqrt'
        else:
            max_f = None

        if param[3] == 0:
            crit = 'gini'
        else:
            crit = 'entropy'

        # create the model
        model = RandomForestClassifier(n_estimators=int(param[0]), max_depth=int(param[1]), max_features=max_f,
                                       criterion=crit, min_impurity_decrease= float(param[4]), ccp_alpha = float(param[5]), oob_score=True, n_jobs=-1)

        # fit the model
        model.fit(Xtrain, ytrain)
        print(model.oob_score_)
        return - model.oob_score_


    opt = GPyOpt.methods.BayesianOptimization(f=objective_function,  # function to optimize
                                              domain=domain,  # box-constrains of the problem
                                              acquisition_type='EI',  # Select acquisition function MPI, EI, LCB
                                              de_duplication=True)
    opt.acquisition.exploration_weight = 0.5

    opt.run_optimization(max_iter=iter - 5)


    x_best = opt.X[np.argmin(opt.Y)]
    print("The best parameters obtained: n_estimators=" + str(x_best[0]) + ", max_depth=" + str(
        x_best[1]) + ", max_features=" + str(
        x_best[2]) + ", criterion=" + str(
        x_best[3]))
    ##########################Comparison###################################
    ## comparison between random search and bayesian optimization
    ## we can plot the maximum oob per iteration of the sequence

    # collect the maximum each iteration of BO, note that it is also provided by GPOpt in Y_Best
    y_bo = 1 - np.maximum.accumulate(-opt.Y).ravel()
    # define iteration number
    xs = range(0,len(y_bo))

    xs_matrix[j, :] =  max_oob_per_iteration
    y_bo_matrix[j, :] = y_bo

    j += 1

xs_average = np.mean(xs_matrix, axis=0)
y_bo_average = np.mean(y_bo_matrix, axis = 0)

plt.plot(xs, xs_average, 'o-', color = 'red', label='Random Search')
plt.plot(xs, y_bo_average, 'o-', color = 'blue', label='Bayesian Optimization')
plt.legend()
plt.xlabel('Iterations')
plt.ylabel('Out of bag error')
plt.title('Comparison between Random Search and Bayesian Optimization')
plt.show()


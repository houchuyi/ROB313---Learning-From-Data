
import numpy as np
import pandas
import math
from matplotlib import pyplot as plt
from data.data_utils import load_dataset
from tqdm import tqdm

datasets = ['pumadyn32nm', 'iris', 'mnist_small', 'mauna_loa', 'rosenbrock']

def loadData(dataset):
    '''
    To load the dataset. Include one exception for the dataset 'rosenbrock'
    '''
    if dataset not in datasets:
        return 0,0,0,0,0,0
    elif dataset == 'rosenbrock':
        xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('rosenbrock', n_train=1000, d=2)
    else:
        xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset(str(dataset))
    return xtrain, xvalid, xtest, ytrain, yvalid, ytest

def L_1_metric(a,b):
    return np.linalg.norm([a-b], ord=1) #calculate the mahattan distance
def L_2_metric(a,b):
    return np.linalg.norm([a-b],ord=2) #calculate the Euclidean distance

def RMSE(predict, target):
    return np.sqrt(np.average(np.abs(predict - target)**2)) #root mean square error

def five_fold_cross_validation(xtrain, xvalid, ytrain, yvalid, distance_metric, k_list=None):

    '''
    GENERAL PROCEDURE:
    1. Shuffle the dataset randomly using seed
    2. Split the dataset into k = 5 groups
    3. For each unique group:
        a. Take the group as a test dataset
        b. Take the remaining (4) groups as a training dataset
        c. Fit the model on the training set and evaluate it on the test dateset
        d. Retain the evaluation score an discard the model
    4. Summarize the model's performance using the sample of model evaluation scores

    return final_errors = [k, metric, error]
    '''

    #merge train and valid datasets first
    x = np.vstack([xtrain,xvalid])
    y = np.vstack([ytrain,yvalid])

    assert (len(x)==len(y)) #execute only when len(x) = len(y), otherwise raise error

    #1
    np.random.seed(5)
    np.random.shuffle(x)
    np.random.seed(5)
    np.random.shuffle(y)

    final_errors = []
    rmse = {} # create a disctionary that contains k as the key and distance_metric as the value

    if k_list == None:
        k_list = list(range(1,21)) # we can play around with this parameter.
        #basically the algorithm will iterate through many ks, and finally find the best one.

    #2
    length = len(x)//5
    #3
    for i in tqdm(range(5), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
        #a, b

        x_test = x[i*length:(i+1)*length]
        x_train = np.vstack([x[:i*length], x[(i+1)*length:]])
        y_test = y[i*length:(i+1)*length]
        y_train = np.vstack([y[:i*length], y[(i+1)*length:]])

        for metric in distance_metric:
            predict = {} # create a dictionary containing k as the key and average y as the value for each k
            for j in tqdm(range(len(x_test)),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
                distances = []
                for a in range(len(x_train)):
                    temp = metric(x_train[a],x_test[j])
                    distances.append((temp,y_train[a]))
            distances.sort(key = lambda x: x[0]) # sort distance based on distances calculated by the metric

            for k in k_list:
                yy = []
                for dist in distances[:k]:
                    yy.append(dist[1]) # add y_train[a] to y
                y_avg = sum(yy)/len(yy)

                if k not in predict:
                    predict[k] = []
                predict[k].append(y_avg)

        for k in k_list:
            if (k ,metric) not in rmse:
                rmse[(k, metric)] = []
            error = RMSE(predict[k], y_test)
            rmse[(k,metric)].append(error)
    for k, metric in rmse:
        error = sum(rmse[(k,metric)])/len(rmse[(k,metric)]) # average all the errors calculated in each interpolation
        #length should be 5 for 5-five_fold_cross_validation
        final_errors.append((k,metric,error))

    return final_errors


def kNN(dataset, regression=True):
    '''
        Takes in one dataset at one instance.
        regression: True for regression, False for classification

        return: the lowest k and the corresponding RMSE

        For Q1, need to store data of the mauna_loa's result

    '''
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = loadData(str(dataset))

    distance_metric = [L_1_metric, L_2_metric]
    if regression:
        result = []
        print("~~~~~~~~~~REGRESSION~~~~~~~~~~")
        print("Dataset:" + dataset)
        #special case for manua_loa, only use l2 metric
        if dataset == 'mauna_loa':
            distance_metric = [L_2_metric]
        result = five_fold_cross_validation(xtrain, xvalid, ytrain, yvalid, distance_metric)
        result.sort(key = lambda x:x[2]) # sort based on the error
        #print the best results
        print("Best K:" + str(result[0][0]) + "  Preferred Distance Metric:" + str(result[0][1]))
        print("RMSE:" + str(result[0][2]))

        # print("###############ALL RESULTS###############")
        # print(result)

    return result[0][0], result[0][1]

#if '__name__' == '__kNN__':
    '''
        For Q1, only apply for regression datasets (i.e. 'pumadyn32nm', 'mauna_loa', 'rosenbrock')
    '''

#REGRESSION
for dataset in ['mauna_loa','rosenbrock', 'pumadyn32nm']:
    k, metric = kNN(dataset, True)
    print(k , metric)

    # #CLASSFICATION
    # for dataset in ['iris', 'mnist_small']:
    #     k, metric = kNN(dataset, False)

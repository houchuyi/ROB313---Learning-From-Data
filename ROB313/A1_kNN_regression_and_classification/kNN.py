
import numpy as np
import pandas
import math
import time
from sklearn import neighbors
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
    return np.sqrt(((predict - target)**2).mean()) #root mean square error

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
    np.random.seed(10)
    np.random.shuffle(x)
    np.random.seed(10)
    np.random.shuffle(y)

    final_errors = []
    rmse = {} # create a disctionary that contains k as the key and distance_metric as the value

    if k_list == None:
        k_list = list(range(1,31)) # we can play around with this parameter.
        #basically the algorithm will iterate through many ks, and finally find the best one.

    #2
    length = len(x)//5
    #3
    for i in tqdm(range(5),desc = 'Fold', ncols = 80):
        #a, b

        x_valid = x[i*length:(i+1)*length]
        x_train = np.vstack([x[:i*length], x[(i+1)*length:]])
        y_valid = y[i*length:(i+1)*length]
        y_train = np.vstack([y[:i*length], y[(i+1)*length:]])

        for metric in distance_metric:
            predict = {} # create a dictionary containing k as the key and average y as the value for each k
            #calculate all the distance from training points to validation points
            for j in tqdm(range(len(x_valid)),desc = 'x_valid', ncols = 80):
                distances = []
                for a in range(len(x_train)):
                    distances.append((metric(x_train[a],x_valid[j]) ,y_train[a]))
                distances.sort(key = lambda x: x[0]) # sort distance based on distances calculated by the metric

                #finding k nearest neighbours
                for k in k_list:
                    yy = 0
                    for dist in distances[:k]:
                        yy += dist[1] # add y_train[a] to y
                    if k not in predict:
                        predict[k] = []
                    predict[k].append(yy/k) #cluster representative value

            for k in k_list:
                if (k ,metric) not in rmse:
                    rmse[(k, metric)] = []
                rmse[(k,metric)].append(RMSE(predict[k], y_valid))

    for k, metric in rmse:
        error = sum(rmse[(k,metric)])/5 # average all the errors calculated in each interpolation
        #length should be 5 for 5-five_fold_cross_validation
        final_errors.append((k,metric,error))

    return final_errors

def test_regression(xtrain, xvalid, xtest, ytrain, yvalid, ytest, k, distance_metric, plot=False):
    '''

    plot: True for mauna_loa, False for others
    '''
    x = np.vstack([xtrain, xvalid])
    y = np.vstack([ytrain, yvalid])

    predict = []
    train_predict = []
    for test in xtest:
        a = []
        for i in range(len(x)):

            a.append((distance_metric(test, x[i]), y[i]))
        a.sort(key=lambda x: x[0])

        #calculate the prediction by kth neighbours
        y_est = 0
        for item in a[:k]:
            y_est += item[1]
        avg = y_est/k

        predict.append(avg)
    for point in xtrain:
        b = []
        for i in range(len(xvalid)):
            b.append((distance_metric(point, xvalid[i]), y[i]))
        b.sort(key=lambda x: x[0])
        y_est = 0
        for item in b[:k]:
            y_est += item[1]
        avg = y_est/k

        train_predict.append(avg)

    test_error = RMSE(ytest, predict)
    if plot:
        # this section is for mauna loa dataset only
        plt.figure(2)
        plt.plot(xtest, ytest, '-g', label='Actual')
        plt.plot(xtest, predict, '-r', label='Prediction')
        plt.xlabel('x test')
        plt.ylabel('y')
        plt.legend(loc='upper right')
        plt.title('Test predictions for Mauna Loa dataset')
        plt.savefig('mauna_loa_test_prediction.png')

    return test_error

def classification(xtrain, xvalid, ytrain, yvalid, distance_metric, k_list=None):
    '''
    Run through k = 1 to 30
    return result containing k-value, corresponding metric and accuracy ratio
    Similar implementation as the five_fold_cross_validation
    Instead of computing RMSE, we simply find the modes
    '''

    np.random.seed(10)
    np.random.shuffle(xtrain)
    np.random.seed(10)
    np.random.shuffle(ytrain)

    if not k_list:
        k_list = list(range(1,31))

    correct = {}            # dictionary with k as keys, and with

    for metric in distance_metric:

        for i in tqdm(range(len(xvalid)), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}'):
            distances = []
            for j in range(len(xtrain)):
                distances.append((metric(xtrain[j],xvalid[i]), ytrain[j]))

            distances.sort(key = lambda x: x[0]) #again, sort the list by dist computed

            #find k closest neighbours
            for k in k_list:
                y = []
                # append all the y values to a vector called y (stored in a dict)
                for dist in distances[:k]:
                    y.append(dist[1])

                # find the most commonly occuring class
                count = {}
                # need to count the y points
                for point in y:
                    if str(point) not in count:
                        count[str(point)] = (point, 0)
                    #for every occurance of a class, we add one to the dict
                    count[str(point)] = (point, count[str(point)][1] + 1)

                # sort the occurences to determine most common
                occurences = list(count.values())
                occurences.sort(key=lambda x: x[1], reverse=True)
                #find the most occurences, thus ust reversed order sort

                prediction = occurences[0][0]

                if np.all(prediction == yvalid[i]):
                    if (k, metric) not in correct:
                        correct[(k, metric)] = 0
                    correct[(k, metric)] += 1

    # for each k value after the 5 folds, calculate the ratio of correctness to total points
    result = []
    for k, metric in correct:
        ratio = correct[(k, metric)]/len(xvalid)   # computes accuracy
        result.append((k, metric, ratio))

    return result

def test_classification(xtrain, xvalid, xtest, ytrain, yvalid, ytest, k, distance_metric):
    '''
    runs the test set using the x_train + x_valid to calculate the estimates

    '''
    x_train = np.vstack([xtrain, xvalid])
    y_train = np.vstack([ytrain, yvalid])

    y_pred = []
    correct = 0

    for i in range(len(xtest)):
        distances = []
        count = {}
        for j in range(len(xtrain)):
            distances.append((distance_metric(xtrain[j], xtest[i]), ytrain[j]))
        distances.sort(key = lambda x: x[0])

        k_dist = distances[:k]

        assert(len(k_dist) == k)

        # need to count the y points
        for dist, point in k_dist:
            if str(point) not in count:
                count[str(point)] = (point, 0)
            count[str(point)] = (point, count[str(point)][1] + 1)

        # sort the occurences to determine most common
        occurences = list(count.values())
        occurences.sort(key=lambda x: x[1], reverse=True)

        prediction= occurences[0][0]

        if np.all(prediction == ytest[i]):
            correct += 1

    # return accuracy (%)
    return correct/len(xtest)

def main(dataset, regression=True):
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
        #special case for mauna_loa, only use l2 metric
        if dataset == 'mauna_loa':
            distance_metric = [L_2_metric]
        result = five_fold_cross_validation(xtrain, xvalid, ytrain, yvalid, distance_metric)
        result.sort(key = lambda x:x[2]) # sort based on the error
        #print the best results
        print("Best K:" + str(result[0][0]) + "  Preferred Distance Metric:" + str(result[0][1]))
        print("RMSE:" + str(result[0][2]))

        # print("###############ALL RESULTS###############")
        # print(result)

        if dataset == 'mauna_loa':
            test_error = test_regression(xtrain, xvalid, xtest, ytrain, yvalid, ytest, result[0][0], result[0][1], plot=True)

            #make the plot vs k
            ks = []
            datas = []
            result.sort(key = lambda x:x[0]) #sort result by k, so we can plot
            for k, metric, err in result:
                ks.append(k)
                datas.append(err)
            plt.figure(1)
            plt.plot(ks, datas)
            plt.xlabel('k')
            plt.ylabel('RMSE')
            plt.title("5-fold cross validation error v.s. different ks for manua loa dataset")
            plt.savefig('mauna_loa_k_rmse.png')

        else:
            test_error = test_regression(xtrain, xvalid, xtest, ytrain, yvalid, ytest, result[0][0], result[0][1], plot=False)
        print("Test Error:" + str(test_error))

    if not regression:
        print("~~~~~~~~~~CLASSFICATION~~~~~~~~~~")
        result = classification(xtrain, xvalid, ytrain, yvalid, distance_metric)
        result.sort(key = lambda x:x[2], reverse=True) # sort by ratio in descending order
        print('Dataset:' + str(dataset))
        print("Best k:" + str(result[0][0]))
        print("Distance Metric:" + str(result[0][1]))
        print("Validation Accuracy:" + str(result[0][2]))

        test_acc = test_classification(xtrain, xvalid, xtest, ytrain, yvalid, ytest, result[0][0], result[0][1])

        print("Test Accuracy:" + str(test_acc))

    return result[0][0], result[0][1]

#if '__name__' == '__kNN__':
    '''
        For Q1, only apply for regression datasets (i.e. 'pumadyn32nm', 'mauna_loa', 'rosenbrock')
    '''

def performance(method,d):
    '''
    This function is only for Question 3, for performance study on kdTree data structure
    '''
    start = time.time()

    xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('rosenbrock', n_train=5000, d=d)

    x = np.vstack([xtrain, xvalid])
    y = np.vstack([ytrain, yvalid])

    np.random.seed(10)
    np.random.shuffle(x)
    np.random.seed(10)
    np.random.shuffle(y)

    k = 5
    metric = L_2_metric

    predictions = []

    if method == 'brute_force': #this part is similar to previous implementation
        #this time, we set k = 5 and distance metric to L2, and find the test rmse value
        #directly.
        try:
            with tqdm(xtest, desc = "Brute Force", ncols = 100) as t:

                for i in t:
                    distances = []
                    for a in range(len(x)):
                        distances.append((metric(x[a],i) ,y[a]))
                    distances.sort(key = lambda x: x[0])
                    yy = 0
                    for pair in distances[:k]:
                        yy += pair[1]
                    predictions.append(yy/k)

        except KeyboardInterrupt:
            t.close()
            raise
        t.close
        test_error = RMSE(ytest, predictions)

    if method == 'kdTree': #kdTree data structure implementation
        kdt = neighbors.KDTree(x)
        dist, kk = kdt.query(xtest,k=5)
        predictions = np.sum(y[kk], axis=1)/k

        test_error = RMSE(predictions, ytest)

    runtime = time.time() - start
    return runtime, test_error


def test_performance():
    d = []
    runtimeB = []
    runtimeT = []
    for i in range(2,11):
        b_runtime, b_test_error = performance('brute_force',i)
        t_runtime, t_test_error = performance('kdTree',i)
        d.append(i)
        runtimeB.append(b_runtime)
        runtimeT.append(t_runtime)
    print("######Brute Force Approach######")
    print("Runtime: " + str(b_runtime))
    print("Test Error: " +str(b_test_error))

    print("######kdTree######")
    print("Runtime: " + str(t_runtime))
    print("Test Error: " +str(t_test_error))

    plt.figure(2)
    plt.plot(d, runtimeB, '-r', label = 'Brute Force')
    plt.plot(d, runtimeT, '-b', label = 'k-d Tree')
    plt.xlabel('d')
    plt.ylabel('Runtime')
    plt.legend(loc='upper right')
    plt.title("Brute Force Runtime & k-d Tree Runtime for varying values of d")
    plt.savefig('runtime_d.png')



def plot_mauna_loa():

    '''
    for plotting the cross validation prediction on serveral different k using L2 metric
    '''
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('mauna_loa')

    klist = [1,2,8,10,20]
    metric = L_2_metric

    x = np.vstack((xtrain, xvalid, xtest))
    y = np.vstack((ytrain, yvalid, ytest))
    rval = [[]]
    predict = [[]]
    j = 0
    for k in klist:

        for test in tqdm(x, ncols = 100):
            a = []
            for i in range(len(x)):
                a.append((metric(test, x[i]), y[i]))
            a.sort(key=lambda x: x[0])

            #calculate the prediction by kth neighbours
            y_est = 0
            for item in a[:k]:
                y_est += item[1]
            avg = y_est/k

            predict[j].append(avg)
        for idx in range(len(x)):
            rval[j].append((x[idx],y[idx],predict[j][idx]))
        rval[j].sort(key = lambda x:x[0])
        rval.append([])

        j += 1
        predict.append([])

    x = [item[0] for item in rval[0]]
    y = [item[1] for item in rval[0]]

    plt.figure(i)
    plt.plot(x, y, label='Actual')
    for w in range(j):
        prediction = [item[2] for item in rval[w]]
        plt.plot(x, prediction, label='Train k=' + str(klist[w]))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(loc='upper right')
    plt.title('Test predictions at serveral k V.S. actual for Mauna Loa dataset')
    plt.savefig('mauna_loa_actual_prediction.png')

#############################Q1#############################
#REGRESSION
# for dataset in ['mauna_loa']: #'mauna_loa','rosenbrock']
#     k, metric = main(dataset, True)
#
#plot_mauna_loa()
#############################Q2#############################
#CLASSFICATION
# for dataset in ['iris', 'mnist_small']:
#     k, metric = train(dataset, False)
#############################Q3#############################
#test_performance()

#############################Q4#############################
#see svd.py

from matplotlib import pyplot as plt
from data.data_utils import load_dataset
from data.data_utils import plot_digit
from tqdm import tqdm
import numpy as np


def sigmoid(z):
    return np.divide(1, np.add(1,np.exp(-z)))
#
# def deri_sigmoid(z):
#     return np.multiply(sigmoid(z),np.subtract(1,sigmoid(z)))

# def log_likelihood(Fhat, y):
#     return np.sum(np.add(np.multiply(y, np.log(Fhat)), np.multiply(np.subtract(1, y), np.log(np.subtract(1, Fhat)))))

def log_likelihood(estimates, actual):
    total = 0
    for i in range(len(estimates)):
        total += actual[i]*np.log(sigmoid(estimates[i])) + (1-actual[i])*np.log(1 - sigmoid(estimates[i]))
    return total

# def likelihood_grad(f_hat, x, y, w):
#     '''
#     computes the gradient of the likelihood P(y|w,X)
#     '''
#     gradient = np.zeros(np.shape(w))
#     v = np.subtract(y, f_hat)
#     for i in range(len(x)):
#         # compute the additive gradient
#         gradient = np.add(gradient, v[i] * x[i])
#     return gradient

def X_matrix(data):
    X = np.ones((len(data), len(data[0]) + 1))
    X[:, 1:] = data
    return X

def compute_accuracy_ratio(y_test, y_estimates):
    return (y_estimates == y_test).sum() / len(y_test)

def Q1_Logistic_Reg_Model(lr = [0.01,0.001,0.0001]):

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:,(1,)], y_valid[:,(1,)], y_test[:,(1,)]
    iter = list(range(10000))
    X = X_matrix(x_train)

    #for plotting
    GD_losses,SGD_losses = [],[]

    #run for different learnging rates
    for r in range(len(lr)):
        #initialize all weights to be zero
        GD_weight = np.zeros(np.shape(X[0,:]))
        SGD_weight = np.zeros(np.shape(X[0,:]))
        GD_loss, SGD_loss = [],[]
        for i in tqdm(iter,desc='iterations', ncols = 60):
            #estimate on training set
            estimate = np.dot(X,GD_weight)
            estimate = estimate.reshape(np.shape(y_train))

            SGD_estimate = np.dot(X,SGD_weight)
            SGD_estimate = SGD_estimate.reshape(np.shape(y_train))
            #GD
            #compute full batch gradient
            GD_gradient = np.zeros(np.shape(GD_weight))
            for m in range(len(y_train)):
                GD_gradient += np.multiply((y_train[m] - sigmoid(estimate[m])), X[m,:])
            GD_weight = np.add(GD_weight, np.multiply(GD_gradient,lr[r]))
            GD_loss.append(-log_likelihood(estimate, y_train)) # NEG LOSS

            #SGD
            index = np.random.randint(0,len(y_train)-1)
            #compute mini batch gradient
            SGD_gradient = np.multiply((y_train[index] - sigmoid(SGD_estimate[index])),X[index,:])
            SGD_weight = np.add(SGD_weight, np.multiply(SGD_gradient, lr[r]))#bs = 1
            SGD_loss.append(-log_likelihood(SGD_estimate, y_train))
            #CAUTION, the weight is the next iteration, and the loss is not plotted

        GD_losses.append(GD_loss)
        SGD_losses.append(SGD_loss)

    # GD_acc = compute_accuracy(GD_weight,X,y_train)
    # SGD_acc = compute_accuracy(SGD_weight,X, y_train)
    # print('Final Accuracy for GD is: ' +str(GD_acc)+'%')
    # print('Final Accuracy for SGD is: ' +str(SGD_acc)+'%')
    #plotting
    for i in range(len(lr)):
        plt.figure(i+1)
        plt.plot(iter, GD_losses[i], '-g', label = 'GD (-)Log_Loss lr=' +str(lr[i]))
        plt.plot(iter, SGD_losses[i], '-r', label = 'SGD (-)Log_Loss lr=' +str(lr[i])+ ',bs=1')
        plt.xlabel('Iterations')
        plt.ylabel('(-) Log_Loss')
        plt.title('Comparison between SGD and GD Training Curves at learning rate = '+str(lr[i]))
        plt.legend(loc = 'best')
        plt.savefig('Q1_comparison_'+str(lr[i])+'_.png')

def Q1_test(lr,method):
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:,(1,)], y_valid[:,(1,)], y_test[:,(1,)]

    #combine tran and valid for test
    x_train = np.vstack([x_train, x_valid])
    y_train = np.vstack([y_train, y_valid])

    X = X_matrix(x_train)
    X_test = X_matrix(x_test)

    iter = list(range(10000))

    #for return
    test_acc = []
    test_logs = []
    neg_log = {}
    test_accuracies = []

    #run for different learnging rates
    for r in range(len(lr)):
        #initialize all weights to be zero
        weight = np.zeros(np.shape(X[0,:]))
        neg_log[lr[r]] = []
        for i in tqdm(iter,desc='iterations', ncols = 60):
            #estimate on training set
            estimate = np.dot(X,weight)
            estimate = estimate.reshape(np.shape(y_train))

            if method == 'GD':
                gradient = np.zeros(np.shape(weight))
                for i in range(len(y_train)):
                    gradient += np.multiply((y_train[i] - sigmoid(estimate[i])), X[i, :])

            elif method == 'SGD':
                m = np.random.randint(0, len(y_train)-1)
                gradient = np.multiply((y_train[m] - sigmoid(estimate[m])), X[m, :])
            #CAUTION, the weight is the next iteration, and the loss is not plotted
            #however, both method converges ok so the next iteration weight will not affect the result too much.
            weight = np.add(weight, lr[r]*gradient)
            neg_log[lr[r]].append(-log_likelihood(estimate,y_train))

        test_estimates = np.dot(X_test, weight)
        test_estimates = test_estimates.reshape(np.shape(y_test))
        predictions = np.zeros(np.shape(y_test))
        for i in range(len(predictions)):
            p = sigmoid(test_estimates[i])
            if p > 1/2:
                predictions[i] = 1
            elif p < 1/2:
                predictions[i] = 0
            else:
                predictions[i] = -1

        test_accuracies.append(compute_accuracy_ratio(y_test, predictions))
        test_logs.append(log_likelihood(test_estimates, y_test))

    #return the best accuracy and its corresponding lr
    best_accuracy = max(test_accuracies)
    test_log = min(test_logs)
    best_lr = []
    best_lr.append(lr[test_accuracies.index(best_accuracy)])
    best_lr.append(lr[test_logs.index(test_log)])
    return neg_log, best_accuracy, best_lr, test_log

comparison = False
test = True

lr = [0.01,0.001,0.0001]

if comparison:
    Q1_Logistic_Reg_Model(lr)

if test:
    neg_log, best_accuracy, best_lr, test_log = Q1_test(lr,'GD')

    print("Logistic Regression Model on dataset: iris")
    print("---GD--- ")
    print("The best test accuracy is "+str(best_accuracy) + "and its corresponding lr = " +str(best_lr[0]))
    print("Test Loss: " +str(test_log) + " and its corresponding lr = "+str(best_lr[1]))

    neg_log, best_accuracy, best_lr, test_log = Q1_test(lr,'SGD')

    print("Logistic Regression Model on dataset: iris")
    print("---SGD--- ")
    print("The best test accuracy is "+str(best_accuracy) + "and its corresponding lr = "+ str(best_lr[0]))
    print("Test Loss: " +str(test_log) + " and its corresponding lr = "+str(best_lr[1]))

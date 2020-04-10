from useful_func import *
from tqdm import tqdm
import numpy as np
import math
import scipy
from matplotlib import pyplot as plt
from data.data_utils import load_dataset

#---------------Q1 PART A---------------#
def a_Laplace_Approx(lr,iter):
    #load data
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]
    x_train, x_test = np.vstack((x_train, x_valid)), x_test
    y_train, y_test = np.vstack((y_train, y_valid)), y_test
    #variance specified from the question
    variances = [0.5,1,2]

    #vectorize
    X_train = X_matrix(x_train)
    X_test = X_matrix(x_test)

    #store for return
    log_marg_ll = {}

    for sigma in variances:
        #initialize zero weight
        weight = np.zeros(np.shape(X_train[0]))

        for i in tqdm(iter,desc='iterations', ncols = 60):

            #compute fhat, graident and weight update
            fhat = np.dot(X_train, weight)
            fhat = np.reshape(fhat, np.shape(y_train))
            gradient = likelihood_grad(X_train, fhat, y_train) + prior_grad(weight, sigma)
            weight = np.add(weight, lr*gradient)

        #compute hessian
        H = likelihood_2grad(X_train, fhat) + prior_2grad(weight, sigma)
        #if sigma ==1: print(weight) #Finds the mean when variance is 1
        #compute log marg ll
        log_marg_ll[sigma] = log_likelihood(fhat, y_train) + log_prior(weight, sigma) - log_g(H)

    return log_marg_ll

#---------------Q1 PART B---------------#
def b_importance_sampling(mean, nsamples, visual=False):
    #load data
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('iris')
    y_train, y_valid, y_test = y_train[:, (1,)], y_valid[:, (1,)], y_test[:, (1,)]

    #make sure ground truth are np arrays with interger entries
    y_train = np.asarray(y_train, int)
    y_valid = np.asarray(y_valid, int)
    y_test = np.asarray(y_test, int)

    # #make x matrices
    X_train = X_matrix(x_train)
    X_valid = X_matrix(x_valid)
    X_test = X_matrix(x_test)

    # #prior variance
    prior_sigma = 1

    #proposal distribution tests
    proposal_var = [1,1.5,2,2.5,3]

    min_ll = np.inf
    for size in tqdm(nsamples,desc='sample_size', ncols=60):
        for variance in tqdm(proposal_var,desc='variance', ncols=60):
            #initialize valid_pred and storage for computing accuracy ratio
            valid_pred = np.zeros(np.shape(y_valid))
            valid_classify =  np.zeros(np.shape(y_valid))

            #take sample weights
            weight = sample_weights(size, mean, variance)

            #compute predictions
            for v in range(len(X_valid)):
                #r over sum_r
                sum_r = 0
                for j in range(size):
                    sum_r += r(np.dot(X_train, weight[j]), y_train, weight[j], prior_sigma, variance, mean)
                #outter sum over i
                pred_post = 0
                for i in range(size):
                    prob_y_star = sigmoid(np.dot(X_valid[v],weight[i]))
                    pred_post += prob_y_star*r(np.dot(X_train, weight[i]), y_train, weight[i], prior_sigma, variance, mean)/sum_r

                #make classifications
                valid_pred[v] = pred_post
                if pred_post > 0.5:
                    valid_classify[v] = 1
                elif pred_post < 0.5:
                    valid_classify[v] = 0
                else:
                    valid_classify[v] = -1

            valid_ll = -compute_log_likelihood(valid_pred, y_valid)/len(x_valid)
            if valid_ll < min_ll:
                min_ll = valid_ll
                min_acc = compute_accuracy_ratio(valid_classify,y_valid)
                optimal_var = variance
                optimal_sample_size = size

    # #Remake matrices for test dataset
    x_train = np.vstack((x_train, x_valid))
    X_train = X_matrix(x_train)
    y_train = np.vstack((y_train, y_valid))

    test_pred = np.zeros(np.shape(y_test))
    test_classify =  np.zeros(np.shape(y_test))

    weight = sample_weights(optimal_sample_size, mean, optimal_var)

    #compute test predictions again using the optimal parameters
    for k in tqdm(range(len(X_test)),desc = 'test', ncols=60):
        #r over sum_r
        sum_r = 0
        for j in range(optimal_sample_size):
            sum_r += r(np.dot(X_train, weight[j]), y_train, weight[j], prior_sigma, optimal_var, mean)
        #outter sum over i
        pred_post = 0
        for i in range(optimal_sample_size):
            prob_y_star = sigmoid(np.dot(X_test[k],weight[i]))
            pred_post += prob_y_star*r(np.dot(X_train, weight[i]), y_train, weight[i], prior_sigma, optimal_var, mean)/sum_r

        #make classifications
        test_pred[k] = pred_post
        if pred_post > 0.5:
            test_classify[k] = 1
        elif pred_post < 0.5:
            test_classify[k] = 0
        else:
            test_classify[k] = -1

    test_acc = compute_accuracy_ratio(test_classify, y_test)
    test_ll = compute_log_likelihood(test_pred, y_test)/len(x_test)

    #visualization. Can be enabled or disabled
    if visual:
        size = 15000
        var = 2.5
        weight = sample_weights(size, mean, var)
        sum_r = 0
        for j in tqdm(range(size),ncols = 60):
            sum_r += r(np.dot(X_train, weight[j]), y_train, weight[j], prior_sigma, var, mean)
        pred_post = list()
        for i in tqdm(range(size),ncols = 60):
            pred_post.append(r(np.dot(X_train, weight[i]), y_train, weight[i], prior_sigma, var, mean)/sum_r)

        posterior_visualization(mean,var, pred_post, weight)
    return -test_ll, test_acc, optimal_var, optimal_sample_size, min_ll, min_acc

def posterior_visualization(mean,variance, posterior, w):
    #credit: Christopher Agia
    for i in range(5):
        weights = list()
        # extract specific component of all weights
        for j in range(len(w)):
            weights.append(w[j][i])

        # zip weights and posterior
        weights, posterior = zip(*sorted(zip(weights, posterior)))

        # set up gaussian
        z = np.polyfit(weights, posterior, 1)
        z = np.squeeze(z)
        p = np.poly1d(z)

        # plot
        w_all = np.arange(min(weights), max(weights), 0.001)
        q_w = scipy.stats.norm.pdf(w_all, mean[i], variance)
        plt.figure(i)
        plt.title("Posterior visualization: q(w) mean=" + str(round(mean[i], 2)) + " var=" + str(variance))
        plt.xlabel("w[" + str(i+1) + "]")
        plt.ylabel("Probability")
        plt.plot(w_all, q_w, '-g', label="Proposal q(w)")
        plt.plot(weights, posterior, 'ob', label="Posterior P(w|X,y)")
        plt.plot(weights, p(weights),"b--")
        plt.legend(loc='upper right')
        plt.savefig("weight_vis_" + str(i) + ".png")

A = True
B = False

if A:
    print("Q1-----Part A--------")
    lr = 0.001
    iter = list(range(1,20001))
    log_marg_ll = a_Laplace_Approx(lr,iter)
    for var in log_marg_ll:
        print('Variance = '+str(var)+', The log marginal likelihood is ' + str(log_marg_ll[var]))

if B:
    print("Q1-----Part B--------")
    mean = [-0.87805271,0.29302957,-1.2347739,0.67815586,-0.89401743] #is found by printing the final weights at variance = 1 in part A
    nsamples = [50,70,100,120,140]
    test_ll, test_acc, optimal_var, optimal_sample_size, valid_ll, valid_acc = b_importance_sampling(mean, nsamples, visual = True)

    print('Proposal Distribution: ' + str(optimal_var))
    print('Sample Size: '+ str(optimal_sample_size))
    print('Test Accuracy: '+str(test_acc))
    print('Test Log-Likelihood: '+ str(test_ll))
    print('Validation Accuracy: '+ str(valid_acc))
    print('Validaiton Log-Likelihood: '+ str(valid_ll))

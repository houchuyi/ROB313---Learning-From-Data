import numpy as np
import math


def compute_accuracy_ratio(y_test, y_estimates):
    return (y_estimates == y_test).sum() / len(y_test)


def X_matrix(data):
    X = np.ones((len(data), len(data[0]) + 1))
    X[:, 1:] = data
    return X

def sigmoid(z):
    return np.divide(1, np.add(1, np.exp(-1*z)))


def log_likelihood(x_prod, y_act):
    log_p = np.dot(y_act.T, np.log(sigmoid(x_prod))) + np.dot(np.subtract(1, y_act).T, np.log(np.subtract(1, sigmoid(x_prod))))
    return log_p


def likelihood_grad(X, x_prod, y_act):
    grad = np.zeros(np.shape(X[0]))
    for i in range(len(x_prod)):
        grad += (y_act[i] - sigmoid(x_prod[i])) * X[i]
    return grad


def likelihood_2grad(X, x_prod):
    hess = np.zeros((len(X[0]), len(X[0])))
    sig_vec = np.multiply(sigmoid(x_prod), sigmoid(x_prod) - 1)
    for i in range(len(x_prod)):
        hess = np.add(hess, sig_vec[i] * np.outer(X[i], X[i].T))
    return hess


def log_prior(w, sigma):
    return -len(w)/2 * np.log(2 * np.pi) - len(w)/2 * np.log(sigma) - 1/(2 * sigma) * np.dot(w.T, w)


def prior_grad(w, sigma):
    return -1/sigma * w


def prior_2grad(w, sigma):
    return -1/sigma * np.eye(len(w))


def log_g(hessian):
    return 1/2 * np.log(np.linalg.det(-1 * hessian)) - len(hessian) / 2 * np.log(2 * np.pi)


def likelihood(x, y):
    likelihood = 1
    for i in range(len(x)):
        likelihood *= (sigmoid(x[i]) ** y[i]) * ((1 - sigmoid(x[i])) ** (1 - y[i]))
    return likelihood


def prior_like(w, variance):
    prior = 1
    for i in range(len(w)):
        prior *= 1 / math.sqrt(2 * math.pi * variance) * math.exp(-(w[i] ** 2) / (2 * variance))
    return prior


def proposal_like(w, proposal_var, mean):
    proposal = 1
    for i in range(len(w)):
        proposal *= 1 / math.sqrt(2 * math.pi * proposal_var) * math.exp(-((mean[i] - w[i]) ** 2) / (2 * proposal_var))
    return proposal


def r(x, y, w, prior_var, proposal_var, mean):
    #Pr(y|w,X)Pr(w)/q(w), q(w): proposal distribution
    return likelihood(x, y) * prior_like(w, prior_var) / proposal_like(w, proposal_var, mean)


def proposal(mean, variance):
    return np.random.multivariate_normal(mean=mean, cov=np.eye(np.shape(mean)[0]) * variance)


def sample_weights(sample_size, mean, variance):
    w = list()
    for i in range(sample_size):
        w.append(proposal(mean, variance))
    return w


def compute_log_likelihood(y_pred, y):
    log_p = np.dot(y.T, np.log(y_pred)) + np.dot(np.subtract(1, y).T, np.log(np.subtract(1, y_pred)))
    return log_p

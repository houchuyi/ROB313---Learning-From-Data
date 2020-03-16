import numpy as np
import math
from matplotlib import pyplot as plt
from data.data_utils import load_dataset
from tqdm import tqdm
import random

all_datasets = ["mauna_loa", "rosenbrock", "iris"]
w =  111.2 #2pi/(-0.922565+0.979061)

######################Data###########################
def loadData(dataset):
    '''
    To load the dataset. Include one exception for the dataset 'rosenbrock'
    '''
    if dataset not in all_datasets:
        return 0,0,0,0,0,0
    elif dataset == 'rosenbrock':
        xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('rosenbrock', n_train=1000, d=2)
    else:
        xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset(str(dataset))
    return xtrain, xvalid, xtest, ytrain, yvalid, ytest
#####################################################

#####################Calculation#####################
def L_2_metric(a,b):
    return np.linalg.norm([a-b],ord=2) #calculate the Euclidean distance

def RMSE(predict, target):
    return np.sqrt(np.average(np.abs(predict - target)**2)) #root mean square error

#####################################################

##################basis functions####################
def x(x):
    return math.sqrt(2)*x
def xsq(x):
    return np.power(x,2)
def xsinwx(x):
    return x*np.sin(w*x)
def xcoswx(x):
    return x*np.cos(w*x)

#####################################################

#####################Question 2######################
def Q2_glm_svd_validation(xtrain, xvalid, ytrain, yvalid, reg_lambda=None):
    '''
    For question 2
    Run SVD for the mauna loa data set, and find the best regularization term lambda
    '''
    if reg_lambda == None:
        reg_lambda = range(0,30)

    basis_functions = [x,xsq,xsinwx,xcoswx] #dont need to include 1

    phi_X = np.ones((len(xtrain),1))
    phi_X_valid = np.ones((len(xvalid),1))

    for f in basis_functions:
        phi_X = np.hstack([phi_X, f(xtrain)])
        phi_X_valid = np.hstack([phi_X_valid, f(xvalid)])

    U, sigma, V = np.linalg.svd(phi_X, full_matrices=True)
    S = np.vstack([np.diag(sigma),np.zeros((len(xtrain)-len(sigma),len(sigma)))])

    best_rmse = np.inf
    best_lambda = -1
    ST_S = np.dot(S.T, S)
    for lam in reg_lambda:
        temp = np.linalg.pinv(ST_S + lam*np.eye(len(ST_S))) #(ST_s + lambda)^-1
        weights = np.dot(V.T, np.dot(temp, np.dot(S.T, np.dot(U.T, ytrain))))
        ypred = np.dot(phi_X_valid, weights)
        cur_rmse = RMSE(ypred, yvalid)

        if cur_rmse < best_rmse:
            best_rmse = cur_rmse
            best_lambda = lam

    return best_lambda


def Q2_glm_svd_test(xtrain, xvalid, xtest, ytrain, yvalid, ytest, lam):
    '''
    run SVD for mauna loa test set using lambda found in validation
    '''

    basis_functions = [x, xsq, xsinwx, xcoswx]
    xtotal = np.vstack([xtrain, xvalid])
    y = np.vstack([ytrain, yvalid])

    np.random.seed(10)
    np.random.shuffle(xtotal)
    np.random.seed(10)
    np.random.shuffle(y)

    phi_X = np.ones((len(xtotal), 1))
    phi_X_test = np.ones((len(xtest), 1))

    for f in basis_functions:
        phi_X = np.hstack([phi_X, f(xtotal)])
        phi_X_test = np.hstack([phi_X_test, f(xtest)])

    U, sigma, V = np.linalg.svd(phi_X, full_matrices=True)
    S = np.vstack([np.diag(sigma),np.zeros((len(xtotal)-len(sigma),len(sigma)))])
    ST_S = np.dot(S.T, S)
    temp = np.linalg.pinv(ST_S + lam*np.eye(len(ST_S))) #(ST_s + lambda)^-1
    weights = np.dot(V.T, np.dot(temp, np.dot(S.T, np.dot(U.T, y))))
    ypred = np.dot(phi_X_test, weights)
    rmse = RMSE(ypred, ytest)

    #plotting
    plt.figure(1)
    plt.plot(xtest, ytest, '-g', label = 'Y True')
    plt.plot(xtest, ypred, '-r', label = 'Y Pred')
    plt.xlabel('X Test')
    plt.ylabel('Y')
    plt.title('GLM Predictions on Mauna Loa for Lambda =' + str(lam))
    plt.legend(loc = 'best')
    plt.savefig('Q2_mauna_loa_glm_pred.png')

    return rmse
#####################################################

#######################Q3############################
def kernel(x,z):
    return (1+x*z)**2+ x*z*math.cos(w*(x-z))
    # return 1 + 2*x*z + x*z*(1+math.cos(w*(x-z)))
def Q3_kernelized_glm(xtrain,xvalid,xtest,ytrain,yvalid,ytest,lam = 16):

    basis_functions = [x, xsq, xsinwx, xcoswx]
    xtotal = np.vstack([xtrain, xvalid])
    ytotal = np.vstack([ytrain, yvalid])

    K = np.empty((len(xtotal), len(xtotal))) # gram matrix, K, symmetric
    computed_k = {}
    for row in range(len(xtotal)):
        for col in range(len(xtotal)):
            if (row, col) in computed_k:
                K[row, col] = computed_k[(row,col)] #no need to compute again due to symmetry
            else:
                temp = kernel(xtotal[row],xtotal[col])
                K[row,col] = temp
                computed_k[(col,row)] = temp #save the value for the symmetric entries

    # cholesky factorization: (K+lam) -> RR^T
    R = np.linalg.cholesky((K+lam*np.eye(len(K))))
    # alpha = (R^TR)^-1y = R^-TR^-1y
    alpha = np.dot(np.dot(np.linalg.inv(R).T, np.linalg.inv(R)), ytotal)
    #ypred = Kmatrix*alpha
    kmatrix = np.empty((len(xtest),len(xtotal)))
    for row in range(len(xtest)):
        temp = np.ndarray((len(xtotal)))
        for col in range(len(xtotal)):
            temp[col] = kernel(xtest[row],xtotal[col])
        kmatrix[row,:] = temp

    ypred = np.dot(kmatrix, alpha)
    rmse = RMSE(ypred, ytest)

    plt.figure(2)
    plt.plot(xtest, ytest, '-g', label='Y True')
    plt.plot(xtest, ypred, '-b', label='Y Pred')
    plt.xlabel('X Test')
    plt.ylabel('Y')
    plt.title('GLM dual perspective on Mauna Loa for lambda=' + str(lam))
    plt.legend(loc='best')
    plt.savefig('Q3_mauna_loa_glm_dual_pred.png')

    return rmse

def plot_kernel():
    '''
    plot the kernel, for visulization purpose
    plot k(0,z) and k(1,z+1) where z in [-0.1,0.1]
    '''
    color = ['g','r']
    for i in range(2):
        z = np.linspace(-0.1+i, 0.1+i, 1000)

        k = np.ndarray((len(z),1))
        for j in range(len(z)):
            k[j] = kernel(i,z[j])


        plt.figure(3+i)
        plt.plot(z,k, '-'+str(color[i]), label='k('+str(i)+', z+' +str(i)+')')
        plt.title('Kernel function over z = [-0.1,0.1]')
        plt.xlabel('z')
        plt.ylabel('kernel')
        plt.legend(loc='best')
        plt.savefig('kernel_plot_'+str(i)+'.png')

#####################################################

#########################Q4##########################
def rbf_gaussian(x,z,theta):
    return math.exp(-(L_2_metric(x, z) ** 2) / theta)

def Q4_RBF_validation(xtrain, xvalid, ytrain, yvalid, regression = True):
    '''
    find the optimal theta, regularization parameter
    '''
    thetas = [0.05,0.1,0.5,1,2]
    regs = [0.001,0.01,0.1,1]
    results = {} #use to store all the results and find the theta and lam with the lowest rmse

    for theta in tqdm(thetas,desc='thetas', ncols = 80):
        K = np.empty((len(xtrain),len(xtrain)))
        computed_k = {}
        for i in range(len(xtrain)):
            for j in range(len(xtrain)):
                if (i, j) in computed_k:
                    K[i, j] = computed_k[(i,j)] #no need to compute again due to symmetry
                else:
                    temp = rbf_gaussian(xtrain[i],xtrain[j], theta)
                    K[i,j] = temp
                    computed_k[(j,i)] = temp #save the value for the symmetric entries


        #make the k matrix so that it can be simply dot with alpha
        Kmatrix = np.empty((len(xvalid),len(xtrain)))
        for i in range(len(xvalid)):
            temp = np.ndarray((len(xtrain)))
            for j in range(len(xtrain)):
                temp[j] = rbf_gaussian(xvalid[i], xtrain[j], theta)
            Kmatrix[i,:] = temp
        for lam in regs:
            #use cholesky factorization
            R = np.linalg.cholesky((K+lam*np.eye(len(K))))
            inv_R = np.linalg.inv(R)
            alpha = np.dot(np.dot(inv_R.T, inv_R),ytrain)

            #store results based on regression and classification
            if regression:
                ypred = np.dot(Kmatrix,alpha)
                rmse = RMSE(ypred, yvalid)
                results[(theta, lam)] = rmse
            else:
                ypred = np.argmax(np.dot(Kmatrix, alpha), axis = 1)
                ytarget = np.argmax(1*yvalid, axis=1)
                results[(theta,lam)] = (ypred==ytarget).sum()/len(ytarget)


    #find the best theta and lambda
    if regression:
        rval = sorted(results.items(), key=lambda x:x[1])
        best_theta = rval[0][0][0]
        best_lambda = rval[0][0][1]
        valid_rmse = rval[0][1]
        return best_theta, best_lambda, valid_rmse
    else:
        rval = sorted(results.items(), key=lambda x:x[1], reverse = True)
        best_theta = rval[0][0][0]
        best_lambda = rval[0][0][1]
        valid_accuracy = rval[0][1]
        return best_theta, best_lambda, valid_accuracy


def Q4_RBF_test(xtrain, xvalid, xtest, ytrain, yvalid, ytest, theta, lam, regression = True):

    xtotal = np.vstack([xtrain, xvalid])
    ytotal = np.vstack([ytrain, yvalid])

    K = np.empty((len(xtotal), len(xtotal))) # gram matrix, K, symmetric
    computed_k = {}
    for row in range(len(xtotal)):
        for col in range(len(xtotal)):
            if (row, col) in computed_k:
                K[row, col] = computed_k[(row,col)] #no need to compute again due to symmetry
            else:
                temp = rbf_gaussian(xtotal[row],xtotal[col], theta)
                K[row,col] = temp
                computed_k[(col,row)] = temp #save the value for the symmetric entries

    Kmatrix = np.empty((len(xtest),len(xtotal)))
    for i in range(len(xtest)):
        temp = np.ndarray((len(xtotal)))
        for j in range(len(xtotal)):
            temp[j] = rbf_gaussian(xtest[i], xtotal[j], theta)
        Kmatrix[i,:] = temp
    #use cholesky factorization
    R = np.linalg.cholesky((K+lam*np.eye(len(K))))
    inv_R = np.linalg.inv(R)
    alpha = np.dot(np.dot(inv_R.T, inv_R),ytotal)

    #store results based on regression and classification
    if regression:
        ypred = np.dot(Kmatrix,alpha)
        result = RMSE(ypred, ytest)
    else:
        ypred = np.argmax(np.dot(Kmatrix, alpha), axis = 1)
        ytarget = np.argmax(1*ytest, axis=1)
        result = (ypred==ytarget).sum()/len(ytarget)

    return result

#####################################################

#########################Q5##########################

def gaussian_kernel(x,z,theta):
    return np.exp(-(np.linalg.norm([x-z],ord=2)**2)/theta)

def MDL(k,N,l2error):
    return (N/2)*math.log(l2error)+(k/2)*math.log(N)

def pick_basis(Icandidates, r, xtrain):
    max = 0
    phi_X = 0
    i = 0
    for j in Icandidates:
        #Iselected.append(j)

        z = xtrain[j]
        phi = np.empty((len(xtrain),1))
        for k in range(len(xtrain)):
            phi[k] = gaussian_kernel(xtrain[k],z,theta)
        Jphi = ((np.dot(phi.T, r))**2)/np.dot(phi.T, phi)
        #want to find i = argmax Jphi
        if max < Jphi:
            max = Jphi
            i = j
            phi_X = phi
        #Iselected.remove(j)
    return i, phi_X

def Q5_orth_match_pursuit(xtrain,xvalid,ytrain,yvalid, theta):
    k = 0 #number of iterations
    # Dict = create_dict_kernel(M,theta) #create the coefficient of the basis functions
    Iselected = []
    Icandidates = list(range(len(xtrain)))
    r = ytrain #first residual, training error
    N = len(xtrain)
    l2error = (np.linalg.norm(r,ord=2)**2)
    pre_l2error = 2*l2error
    weight, pre_weight = 0,0
    phi_X = np.empty((xtrain.shape[0],0))

    while MDL(k-1,N,pre_l2error) > MDL(k,N,l2error):
        pre_l2error = l2error
        pre_weight = weight

        k = k + 1
        i, phi = pick_basis(Icandidates,r, xtrain)
        # math.sqrt(math.pow(2,i)/math.factorial(i))
        Iselected.append(i)
        Icandidates.remove(i)

        phi_X = np.hstack([phi_X, phi])

        #########################################################
        U, sigma, V = np.linalg.svd(phi_X, full_matrices=True)
        S = np.vstack([np.diag(sigma),np.zeros((len(xtrain)-len(sigma),len(sigma)))])
        ST_S = np.dot(S.T, S)
        temp = np.linalg.pinv(ST_S) #(ST_s + lambda)^-1 + lam*np.eye(len(ST_S))
        weight = np.dot(V.T, np.dot(temp, np.dot(S.T, np.dot(U.T, ytrain))))
        ypred = np.dot(phi_X, weight)
        r = ytrain - ypred
        l2error = (np.linalg.norm(r,ord=2)**2)
        ##########################################################

        print(k)
    return Iselected[:-1], pre_weight, MDL(k-1,N,pre_l2error)

def Q5_test(xtrain, xtest, ytrain, ytest, theta, Iselected, weight):
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('rosenbrock', n_train=200, d=2)
    phi_X = np.empty((len(xtest),len(Iselected)))
    for row in range(len(xtest)):
        temp = np.ndarray((len(Iselected)))
        for col in range(len(Iselected)):
            temp[col] = gaussian_kernel(xtest[row],xtrain[col],theta)
        phi_X[row,:] = temp

    ypred = np.dot(phi_X, weight)
    rmse = RMSE(ypred, ytest)

    return rmse
#####################################################

#####################################################
q2 = False
q3 = False
q4 = False
q5 = True

if q2:
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = loadData('mauna_loa')
    best_lambda = Q2_glm_svd_validation(xtrain,xvalid,ytrain,yvalid) # lambda = 14
    q2rmse = Q2_glm_svd_test(xtrain,xvalid,xtest,ytrain,yvalid,ytest,best_lambda)
    print('Q2 The optimal regularization term, lambda, is ' +str(best_lambda)+ ' and it produces the min rmse = '+str(q2rmse))
if q3:
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = loadData('mauna_loa')
    best_lambda = 14 # computed in q2
    q3rmse = Q3_kernelized_glm(xtrain,xvalid,xtest,ytrain,yvalid,ytest,best_lambda)
    print('Q3 The optimal regularization term, lambda, is ' +str(best_lambda)+ ' and it produces the min rmse = '+str(q3rmse))
    plot_kernel()
if q4:
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = loadData('iris')
    best_theta, best_lambda, valid_acc = Q4_RBF_validation(xtrain, xvalid, ytrain, yvalid, regression = False)
    test_acc = Q4_RBF_test(xtrain, xvalid, xtest, ytrain, yvalid, ytest, best_theta, best_lambda, regression = False)
    print('Q4: '+ 'iris')
    print('Optimial Theta=' +str(best_theta) +' Optimal Lambda= '+ str(best_lambda))
    print('Validation Accuracy: ' + str(valid_acc))
    print('Test Accuracy: '+str(test_acc))


    xtrain, xvalid, xtest, ytrain, yvalid, ytest = loadData('rosenbrock')
    best_theta, best_lambda, valid_rmse = Q4_RBF_validation(xtrain, xvalid, ytrain, yvalid,True)
    test_rmse = Q4_RBF_test(xtrain, xvalid, xtest, ytrain, yvalid, ytest, best_theta, best_lambda,True)
    print('Q4: '+ 'rosenbrock')
    print('Optimial Theta=' +str(best_theta) +' Optimal Lambda= '+ str(best_lambda))
    print('Validation RMSE: ' + str(valid_rmse))
    print('Test RMSE: '+str(test_rmse))

    xtrain, xvalid, xtest, ytrain, yvalid, ytest = loadData('mauna_loa')
    best_theta, best_lambda, valid_rmse = Q4_RBF_validation(xtrain, xvalid, ytrain, yvalid,True)
    test_rmse = Q4_RBF_test(xtrain, xvalid, xtest, ytrain, yvalid, ytest, best_theta, best_lambda,True)
    print('Q4: '+ 'mauna_loa')
    print('Optimial Theta=' +str(best_theta) +' Optimal Lambda= '+ str(best_lambda))
    print('Validation RMSE: ' + str(valid_rmse))
    print('Test RMSE: '+str(test_rmse))
if q5:
    xtrain, xvalid, xtest, ytrain, yvalid, ytest = load_dataset('rosenbrock', n_train=200, d=2)
    thetas = [1.0]
    for theta in thetas:
        Iselected, weight, l2loss = Q5_orth_match_pursuit(xtrain,xvalid,ytrain,yvalid, theta)
        print("The final MDL is " + str(l2loss) + ", with theta = "+str(theta))
        Q5_rmse = Q5_test(xtrain, xtest, ytrain, ytest, theta, Iselected,weight)
        print("The result of OMP on the test dataset of rosenbrock:")
        print("Number of basis function used: "+str(len(Iselected)))
        print("Theta: "+str(theta))
        print("Test RMSE: " + str(Q5_rmse))
###########For viewing the data###################
# data = loadData("mauna_loa")
# plt.plot(data[0],data[3])
# plt.show()

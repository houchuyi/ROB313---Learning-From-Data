from a3_mod import *
from matplotlib import pyplot as plt
from data.data_utils import load_dataset
from data.data_utils import plot_digit
from tqdm import tqdm


def Part_C_training(lr,plot = True):
    # load the MNIST_small dataset

    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    # initialize the weights and biases of the network
    M = 100 # neurons per hidden layer
    W1 = np.random.randn(M, 784)/np.sqrt(784) # weights of first (hidden) layer
    W2 = np.random.randn(M, M)/np.sqrt(M)# weights of second (hidden) layer
    W3 = np.random.randn(10, M)/np.sqrt(M) # weights of third (output) layer
    b1 = np.zeros((M, 1)) # biases of first (hidden) layer
    b2 = np.zeros((M, 1)) # biases of second (hidden) layer
    b3 = np.zeros((10, 1)) # biases of third (output) layer
    epoch = 50
    iter = list(range(1,40*epoch+1))
    bs = 250
    train_nll,valid_nll = [],[]
    min_nll = np.inf
    for i in tqdm(iter,desc='iterations', ncols = 60):

        # compute list of 250 random integers (mini-batch indices)
        idx = np.random.choice(x_train.shape[0], size=bs, replace=False)
        mini_batch_x = x_train[idx, :]
        mini_batch_y = y_train[idx, :]

        (nll, (W1_grad, W2_grad, W3_grad, b1_grad, b2_grad, b3_grad)) = nll_gradients(W1, W2, W3, b1, b2, b3, mini_batch_x,mini_batch_y)
        train_nll.append(nll/bs) #loss need to divided by the bs.

        nll_valid = negative_log_likelihood(W1, W2, W3, b1, b2, b3, x_valid, y_valid)
        valid_nll.append(nll_valid/len(x_valid))

        #track the minimum validation loss and its parameters
        if nll_valid < min_nll:
            opt_W1 = W1
            opt_W2 = W2
            opt_W3 = W3
            opt_b1 = b1
            opt_b2 = b2
            opt_b3 = b3
            min_nll = nll_valid
            min_iter = i
        #update weights and biases
        W1 -= lr*W1_grad
        W2 -= lr*W2_grad
        W3 -= lr*W3_grad
        b1 -= lr*b1_grad
        b2 -= lr*b2_grad
        b3 -= lr*b3_grad

        #iterate



    #plotting
    if plot:
        train_acc = compute_accuracy(opt_W1, opt_W2, opt_W3, opt_b1, opt_b2, opt_b3, x_train, y_train)
        valid_acc = compute_accuracy(opt_W1, opt_W2, opt_W3, opt_b1, opt_b2, opt_b3, x_valid, y_valid)
        print('BEST TRAINING ACCURACY: '+str(train_acc*100) +'%')
        print('BEST VALIDATION ACCURACY: '+str(valid_acc*100) +'%')

        print('Final training NLL: ' + str(train_nll[-1]))
        print('Final validation NLL: '+ str(valid_nll[-1])+' Best: ' + str(valid_nll[min_iter]))

        plt.figure(1)
        if lr != 0.001:
            color = ['-g','-r']
        else:
            color = ['-b','y']
        plt.plot(iter, train_nll, color[0], label = 'Training NLL lr = ' + str(lr))
        plt.plot(iter, valid_nll, color[1], label = 'Validation NLL lr = ' + str(lr))
        plt.xlabel('Iterations')
        plt.ylabel('Negative Log Likelihood')
        plt.title('SGD Training (Batch_size = 250) and Validation Curve')
        plt.legend(loc = 'best')
        plt.savefig('Part_C'+str(epoch)+'.png')

    return opt_W1,opt_W2,opt_W3,opt_b1,opt_b2,opt_b3,min_iter


def compute_accuracy(W1, W2, W3, b1, b2, b3, x, y):
    Fhat = np.exp(forward_pass(W1, W2, W3, b1, b2, b3, x))#make it a softmax
    Fhat = np.argmax(Fhat, axis=1)
    y = np.argmax(y, axis=1)
    return (Fhat == y).sum() / len(y)


def Part_D_plot_test(W1,W2,W3,b1,b2,b3,threshold):
    '''
    plot a few test set digits that the top class conditional probability is below a certain threshold
    '''
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_dataset('mnist_small')
    Fhat = np.exp(forward_pass(W1, W2, W3, b1, b2, b3, x_test))#make it a softmax
    Fhat = np.max(Fhat, axis=1)#find the max class conditional probability
    counter = 0
    for i in range(len(Fhat)):
        if Fhat[i]<threshold:
            plot_digit(x_test[i])
            print("True digit: " +str(np.argmax(y_test[i])))
            counter += 1
        if counter >= 5:
            break

    print("Digits are shown...END")
part_c = False
part_d = True
if part_c:
    lr = 0.0002
    W1,W2,W3,b1,b2,b3,min_iter = Part_C_training(lr,True)
    print("The iteration number for the least loss is: " + str(min_iter))

    lr = 0.001
    W1,W2,W3,b1,b2,b3,min_iter = Part_C_training(lr)
    print("The iteration number for the least loss is: " + str(min_iter))

if part_d:
    lr = 0.001
    W1,W2,W3,b1,b2,b3,min_iter = Part_C_training(lr,False)
    Part_D_plot_test(W1,W2,W3,b1,b2,b3,0.5)

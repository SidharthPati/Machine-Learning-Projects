import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import os


def initializeWeights(n_in, n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
       
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""

    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)
    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon
    return W


def sigmoid(z):
    """# Notice that z can be a scalar, a vector or a matrix
    # return the sigmoid of input z"""

    return  # your code here


def preprocess():
    """ Input:
     Although this function doesn't have any input, you are required to load
     the MNIST data set from file 'mnist_all.mat'.

     Output:
     train_data: matrix of training set. Each row of train_data contains 
       feature vector of a image
     train_label: vector of label corresponding to each image in the training
       set
     validation_data: matrix of training set. Each row of validation_data 
       contains feature vector of a image
     validation_label: vector of label corresponding to each image in the 
       training set
     test_data: matrix of training set. Each row of test_data contains 
       feature vector of a image
     test_label: vector of label corresponding to each image in the testing
       set

     Some suggestions for preprocessing step:
     - feature selection"""

    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

    # Split the training sets into two sets of 50000 randomly sampled training examples and 10000 validation examples. 
    # Your code here.
    # print(mat['test0'].shape)
    # Column size obtained: 784

    # Arrays containing data before randomization and normalization --> Section (1)
    train_data_initial = np.zeros((50000, 784))
    train_label_initial = np.zeros(50000)
    validation_data_initial = np.zeros((10000, 784))
    validation_label_initial = np.zeros(10000)
    test_data_initial = np.zeros((10000, 784))
    test_label_initial = np.zeros(10000)

    # Helper variables
    train_data_length = 0
    train_label_length = 0
    validation_data_length = 0
    validation_label_length = 0
    test_data_length = 0

    # Iterate through the data set and split it into arrays in Section (1)
    for key in mat.keys():
        # if training data is encountered
        if 'train' in key:
            val = mat.get(key)
            val_range = range(val.shape[0])
            # randomize val_range
            randomized_val = np.random.permutation(val_range)
            # fetch label which is at the end of the string. Ex: train0, train1 etc.
            label = key[-1]
            # Length of current training set
            val_length = len(val)
            train_val_length = val_length - 1000

            # Get the initial training data
            train_data_initial[train_data_length: train_data_length+train_val_length] = val[randomized_val[1000:], :]
            train_data_length += train_val_length

            # Map the above data to corresponding labels
            train_label_initial[train_label_length: train_label_length+train_val_length] = label
            train_label_length += train_val_length

            # Get the validation data set
            validation_data_initial[validation_data_length:validation_data_length+1000] = val[randomized_val[:1000], :]
            validation_data_length += 1000

            # Map the above data to corresponding labels
            validation_label_initial[validation_label_length: validation_label_length+1000] = label
            validation_label_length += 1000
        # If testing data is encountered
        elif 'test' in key:
            val = mat.get(key)
            val_range = range(val.shape[0])
            # randomize val_range
            randomized_val = np.random.permutation(val_range)
            # fetch label which is at the end of the string. Ex: train0, train1 etc.
            label = key[-1]
            # Length of current training set
            test_val_length = len(val)

            # Get the testing data set
            test_data_initial[test_data_length: test_data_length+test_val_length] = val[randomized_val]
            # Map the above data to corresponding labels
            test_label_initial[test_data_length: test_data_length+test_val_length] = label
            test_data_length += test_val_length

    # Normalized training data to be returned
    train_size = range(train_data_initial.shape[0])
    randomized_train = np.random.permutation(train_size)
    train_data = train_data_initial[randomized_train]
    train_data = np.double(train_data)
    train_data = train_data/255.0
    train_label = train_label_initial[randomized_train]

    # Normalized validation data to be returned
    validation_size = range(validation_data_initial.shape[0])
    randomized_validation = np.random.permutation(validation_size)
    validation_data = validation_data_initial[randomized_validation]
    validation_data = np.double(validation_data)
    validation_data = validation_data/255.0
    validation_label = validation_data_initial[randomized_validation]

    # Normalized testing data to be returned
    test_size = range(test_data_initial.shape[0])
    randomized_test = np.random.permutation(test_size)
    test_data = test_data_initial[randomized_test]
    test_data = np.double(test_data)
    test_data = test_data/255.0
    test_label = test_label_initial[randomized_test]

    # Feature selection
    # Your code here.

    print('preprocess done')

    return train_data, train_label, validation_data, validation_label, test_data, test_label


def nnObjFunction(params, *args):
    """% nnObjFunction computes the value of objective function (negative log 
    %   likelihood error function with regularization) given the parameters 
    %   of Neural Networks, thetraining data, their corresponding training 
    %   labels and lambda - regularization hyper-parameter.

    % Input:
    % params: vector of weights of 2 matrices w1 (weights of connections from
    %     input layer to hidden layer) and w2 (weights of connections from
    %     hidden layer to output layer) where all of the weights are contained
    %     in a single vector.
    % n_input: number of node in input layer (not include the bias node)
    % n_hidden: number of node in hidden layer (not include the bias node)
    % n_class: number of node in output layer (number of classes in
    %     classification problem
    % training_data: matrix of training data. Each row of this matrix
    %     represents the feature vector of a particular image
    % training_label: the vector of truth label of training images. Each entry
    %     in the vector represents the truth label of its corresponding image.
    % lambda: regularization hyper-parameter. This value is used for fixing the
    %     overfitting problem.
       
    % Output: 
    % obj_val: a scalar value representing value of error function
    % obj_grad: a SINGLE vector of gradient value of error function
    % NOTE: how to compute obj_grad
    % Use backpropagation algorithm to compute the gradient of error function
    % for each weights in weight matrices.

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % reshape 'params' vector into 2 matrices of weight w1 and w2
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit j in input 
    %     layer to unit i in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit j in hidden 
    %     layer to unit i in output layer."""

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0

    # Your code here
    #
    #
    #
    #
    #



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural
    % Network.

    % Input:
    % w1: matrix of weights of connections from input layer to hidden layers.
    %     w1(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % w2: matrix of weights of connections from hidden layer to output layers.
    %     w2(i, j) represents the weight of connection from unit i in input 
    %     layer to unit j in hidden layer.
    % data: matrix of data. Each row of this matrix represents the feature 
    %       vector of a particular image
       
    % Output: 
    % label: a column vector of predicted labels"""

    labels = np.array([])
    # Your code here

    return labels


"""**************Neural Network Script Starts here********************************"""

train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
print("Pre processing done")
print("Training data: ")
print(train_data.shape)
print(train_data)
print("Training labels: ")
print(train_label)
print("Validation data: ")
print(validation_data.shape)
print(validation_data)
print("Validation labels: ")
print(validation_label)
print("Test data: ")
print(test_data.shape)
print(test_data)
print("Test labels: ")
print(test_label)
file_name = os.path.expanduser('~/Desktop/preprocessing_output.txt')
np.savetxt(file_name, train_data)


#  Train Neural Network

# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]

# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 50

# set the number of nodes in output unit
n_class = 10

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden)
initial_w2 = initializeWeights(n_hidden, n_class)

# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)

# set the regularization hyper-parameter
lambdaval = 0

args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example

opts = {'maxiter': 50}  # Preferred value.

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal
# and nnObjGradient. Check documentation for this function before you proceed.
# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)


# Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

# Test the computed parameters

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

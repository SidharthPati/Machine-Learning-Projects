
# coding: utf-8

# In[10]:


import numpy as np
import scipy.io as spio
from scipy.optimize import minimize
from scipy.io import loadmat
from math import sqrt
import pickle
import time
import matplotlib.pyplot as plt


# In[2]:


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


# In[3]:


def sigmoid(z):
    return np.divide(1.0, 1.0 + np.exp(-1.0*z))


# In[4]:


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

    # Reference for preprocess() - Prof. Varun Chandola's Lecture - https://youtu.be/XEZbGGqn4yA
    mat = spio.loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary

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
    validation_label = validation_label_initial[randomized_validation]

    # Normalized testing data to be returned
    test_size = range(test_data_initial.shape[0])
    randomized_test = np.random.permutation(test_size)
    test_data = test_data_initial[randomized_test]
    test_data = np.double(test_data)
    test_data = test_data/255.0
    test_label = test_label_initial[randomized_test]

    # Feature selection function call
    
    #     print('preprocess done')
    #     print(train_data.shape)
    #     print(train_label.shape)
    #     print(validation_data.shape)
    #     print(validation_label.shape)
    #     print(test_data.shape)
    #     print(test_label.shape)
    train_data, validation_data, test_data, selected_features = featureSelection(train_data, validation_data, test_data)
    return train_data, train_label, validation_data, validation_label, test_data, test_label, selected_features


# In[5]:


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
    #feedforward propogation
    #adding bias to the input layer
    # Adding Bias to training data
    temp_trainData=np.concatenate((np.transpose(training_data),np.ones(shape=(1,np.transpose(training_data).shape[1]), dtype=int))) # Adding bias
    # print(temp_trainData)
    #Eq 1 in Assignment doc
    input_hidden=np.dot(w1,temp_trainData) 
    # Eq 2 in Assignment doc 
    input_hidden_output=sigmoid(input_hidden)
    input_hidden_output_bias=np.ones(shape=(1,input_hidden_output.shape[1]), dtype=int) # Bias in hidden layer
    input_hidden_output=np.concatenate((input_hidden_output,input_hidden_output_bias)) # Add bias
    # Eq 3 from Assignment doc
    hidden_output=np.dot(w2,input_hidden_output)
    # Eq 4 from Assignment doc
    #dimension of output_layer_output=
    output_sigmoid=sigmoid(hidden_output)
       
    #basically create vector of 50000*10 indicating one wherever the label is true
    # training_label_size = np.size(training_label)
    # print(training_label_size)
    temp_list = np.zeros((training_label.shape[0], n_class), int)
    i = 0
    for train_label in training_label:
        temp_list[i][int(train_label)] = 1
        i += 1
    labels = temp_list.transpose()#10*50000
    
    #Eq 5 from Assignment doc to calculate log likelihood of error
    error = labels*np.log(output_sigmoid) + (1 - labels)*np.log(1 - output_sigmoid) 
    #output_layer is the output from equation 4
    #Eq 6 from Assignment doc
    count=training_data.shape[0]
    lle = (-1)*(np.sum(error[:])/count)
    #Eq 8 from Assignment doc
    derivative_error= np.dot((output_sigmoid - labels),input_hidden_output.transpose())
    #Eq 12 from Assignment doc
    derivative_err_func_hidden =  np.dot(w2.transpose(), 
                                         output_sigmoid - labels)*(input_hidden_output*(1 - input_hidden_output))
    
    #removing the bias
    w1_error = np.dot(derivative_err_func_hidden , temp_trainData.transpose())[:-1,:]
    
    #Eq 13 from Assignment doc
    obj_val = lle + ((np.sum(w1**2)+np.sum(w2**2))/(2*count))*lambdaval
    # print("obj val",obj_val)
    
    #Eq 16 from Assignment doc
    grad1 = (w1_error + lambdaval*w1)/count
    grad2 = (derivative_error + lambdaval*w2)/count 
    
    # print(input_hidden)
    # print(input_hidden_output_bias)
    # print(input_hidden_output)
    # print(output_sigmoid)
    # print(output_sigmoid.shape)

    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)
    obj_grad = np.array([])
    # Reference for flatten: https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.flatten.html
    obj_grad = np.concatenate((grad1.flatten(),grad2.flatten()), 0)
    
    return (obj_val, obj_grad)


def featureSelection(train_data, validation_data, test_data):
    """
    Method to remove redundant features. We take help of set() to fetch unique values in a list.
    When the length of the set is 1 or less, then we eliminate these values
    """
    selected_features = list()
    repeating_vals = list()
    for index_iter in range(train_data.shape[1]):
        temp_list = list(train_data[:,index_iter])
        set1 = set(temp_list)
        if len(set1) <= 1:
            repeating_vals.append(index_iter)
        else:
            selected_features.append(index_iter)
    # Delete rows found in repeating values list from training data, validation data and test data
    # This completes Feature selection
    train_data=np.delete(train_data,repeating_vals,axis=1)
    validation_data=np.delete(validation_data,repeating_vals,axis=1)
    test_data=np.delete(test_data,repeating_vals,axis=1)
    
    return train_data, validation_data, test_data, selected_features


# In[13]:


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
    # Feed forward begins
    training_bias=np.ones(shape=(1,np.transpose(data).shape[1]), dtype=int)
    training_data=np.concatenate((np.transpose(data),training_bias))
    # print(training_data)
    # Neural Network Logic begins here
    #Eq 2 from Assignment doc
    hidden_output=sigmoid(np.dot(w1,training_data))
    hidden_layer_bias=np.ones(shape=(1,hidden_output.shape[1]), dtype=int)
    # Adding bias
    hidden_layer_output=np.concatenate((hidden_output, hidden_layer_bias))
    #Eq 4 from Assignment doc
    output_layer_output=sigmoid(np.dot(w2,hidden_layer_output))
    # Feed forward done
    #     print("inside nnpredict method. print output after feed forward")
    #     print("training_data")
    #     print(training_data)
    #     print("hidden_layer_output")
    #     print(hidden_layer_output)
    #     print("output_layer_output")
    #     print(output_layer_output)
    labels = np.argmax(output_layer_output,axis=0)
    return labels


# In[16]:


train_data, train_label, validation_data, validation_label, test_data, test_label, final_features_list = preprocess()
print("Pre Processing done!")
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
lambdaval = 15

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
# print("before calling predict function")
# print(w1)
# print(w2)
# print(train_data)

predicted_label = nnPredict(w1, w2, train_data)

# find the accuracy on Training Dataset

print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, validation_data)

# find the accuracy on Validation Dataset

print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')

predicted_label = nnPredict(w1, w2, test_data)

# find the accuracy on Validation Dataset

print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

pickle_list = [final_features_list, 50, w1, w2, 15]
pickle.dump(pickle_list, open('params.pickle', 'wb'))

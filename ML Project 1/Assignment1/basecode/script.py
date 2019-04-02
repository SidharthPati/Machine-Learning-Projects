#!/usr/bin/env python
# coding: utf-8

# In[6]:


# %load script.py
import numpy as np
from scipy.optimize import minimize
from scipy.io import loadmat
from numpy.linalg import det, inv
from math import sqrt, pi
import scipy.io
import matplotlib.pyplot as plt
import pickle
import sys

def ldaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmat - A single d x d learnt covariance matrix 
    
    # IMPLEMENT THIS METHOD 
    
    # means =1/N*( i=1 to N Xi)
    #covariance= 1/N*(sum i=1 to n(xi-u)^2
    
    X_dimensions= np.shape(X)[1]
    classLabels = int(np.max(y))
    means = np.zeros((X_dimensions, int(classLabels)))
    len(means)
    # we will be creating data cluster per label and calculate mean values of every cluster
    for i in range (0, classLabels):
        c = np.where(y==i+1)[0]
        dataPerLabel = X[c,:]
        means[:, i] = np.mean(dataPerLabel, axis=0)
    print(means)
    #we need lot of data to compute reliable covariance so in LDA we compute single variance
    covmat = np.cov(np.transpose(X))
    print(covmat)
    return means,covmat

def qdaLearn(X,y):
    # Inputs
    # X - a N x d matrix with each row corresponding to a training example
    # y - a N x 1 column vector indicating the labels for each training example
    #
    # Outputs
    # means - A d x k matrix containing learnt means for each of the k classes
    # covmats - A list of k d x d learnt covariance matrices for each of the k classes
    
    # IMPLEMENT THIS METHOD
    covmats=[]
    X_dimensions= np.shape(X)[1]
    classLabels = int(np.max(y))
    means = np.zeros((X_dimensions, int(classLabels)))
    # we will be creating data cluster per label and calculate mean and variance values of every cluster
    for i in range (0, classLabels):
        c = np.where(y==i+1)[0]
        dataPerLabel = X[c,:]
        means[:, i] = np.mean(dataPerLabel, axis=0)
        covmats.append(np.cov(np.transpose(dataPerLabel)))
    print(means)
    print(covmats)
    return means,covmats

def ldaTest(means,covmat,Xtest,ytest):
    # Inputs
    # means, covmat - parameters of the LDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
    #we are computing Exp^(-0.5(x-meani)*E^-1*(x-meani)). whichever clas label generates maximum value, we assign that class to the data sample
    #the denominator part is just a constant same for all labels since we are dealing with a single variance value
    count=0;
    invCovarianceMat = np.linalg.inv(covmat);
    N = np.shape(Xtest)[0];
    ypred = np.zeros(N);
    classLabels = np.shape(means)[1]
    for i in range (1, N + 1):
        pdf = 0;
        classPrediction = 0;
        testData = np.transpose(Xtest[i-1,:]);
        for k in range (1, classLabels+1):
            res = np.exp((-1/2)*np.dot(np.dot(np.transpose(testData - means[:, k-1]),invCovarianceMat),(testData - means[:, k-1])));
            if (res > pdf):
                classPrediction = k;
                pdf = res;
        ypred[i-1]=classPrediction;
        if (classPrediction == ytest[i-1]):
            count = count + 1;
    acc = count/N;
    print(acc)
    print(ypred)
    return acc,ypred

def qdaTest(means,covmats,Xtest,ytest):
    # Inputs
    # means, covmats - parameters of the QDA model
    # Xtest - a N x d matrix with each row corresponding to a test example
    # ytest - a N x 1 column vector indicating the labels for each test example
    # Outputs
    # acc - A scalar accuracy value
    # ypred - N x 1 column vector indicating the predicted labels

    # IMPLEMENT THIS METHOD
   
   
    count=0;
    N = np.shape(Xtest)[0];
    ypred = np.zeros(N);
    classLabels = np.shape(means)[1]
    DenominatorTerms=np.zeros(classLabels)
    inverseCovmatrix=np.copy(covmats)
    
    #calculating inverse of variance matrix
    for i in range (1, classLabels+1):
        inverseCovmatrix[i-1]=np.linalg.inv(covmats[i-1]);
     #calculating  denominator= 1/((2*pi)^D/2)* |E|^1/2 for every covariance matrix 
    for i in range (1, classLabels+1):
        dimension= np.shape(covmats[i-1])[0];
        DenominatorTerms[i-1] = 1.0/(np.power(2*np.pi,dimension/2)*np.power(np.linalg.det(covmats[i-1]),1/2));
    #print(DenominatorTerms)
    
   
    for i in range (1, N + 1):
        pdf = 0;
        classPrediction = 0;
        testData = np.transpose(Xtest[i-1,:]);
        for k in range (1, classLabels+1):
            #calculating term numerator=denominator* exp^(-0.5 (x-mean) E^-1 (x-mean))
            res = DenominatorTerms[k-1]*np.exp((-1/2)*np.dot(np.dot(np.transpose(testData - means[:, k-1]),inverseCovmatrix[k-1]),(testData - means[:, k-1])));
            #print(res)
            if (res > pdf):
                classPrediction = k;
                pdf = res;
        ypred[i-1]=classPrediction
        if (classPrediction == ytest[i-1]):
            count = count + 1;
    acc = count/N;
    print(acc)
    return acc,ypred

def learnOLERegression(X,y):
    # Inputs:                                                         
    # X = N x d 
    # y = N x 1                                                               
    # Output: 
    # w = d x 1 
    # IMPLEMENT THIS METHOD 
    Xtrans=np.transpose(X)
    operandOne=np.matmul(Xtrans,X)
    invOpOne=np.linalg.inv(operandOne)
    operandTwo=np.matmul(Xtrans,y)
    w=np.matmul(invOpOne,operandTwo)                                                
    return w

def learnRidgeRegression(X,y,lambd):
    # Inputs:
    # X = N x d                                                               
    # y = N x 1 
    # lambd = ridge parameter (scalar)
    # Output:                                                                  
    # w = d x 1                                                                

    # IMPLEMENT THIS METHOD 
    I=np.identity(X.shape[1])
    Xtrans=np.transpose(X)
    operandOne=np.matmul(Xtrans,X)+lambd*I
    invOpOne=np.linalg.inv(operandOne)
    operandTwo=np.matmul(Xtrans,y)
    w=np.matmul(invOpOne,operandTwo)
    return w

def testOLERegression(w,Xtest,ytest):
    # Inputs:
    # w = d x 1
    # Xtest = N x d
    # ytest = X x 1
    # Output:
    # mse
    
    # IMPLEMENT THIS METHOD
    row=len(Xtest)
    product=np.matmul(np.transpose(ytest-np.matmul(Xtest,w)),ytest-(np.matmul(Xtest,w)))
    mse=product/row
    return mse

def regressionObjVal(w, X, y, lambd):

    # compute squared error (scalar) and gradient of squared error with respect
    # to w (vector) for the given data X and y and the regularization parameter
    # lambda                                                                  

    # IMPLEMENT THIS METHOD 
    # Reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.asmatrix.html
    w = np.asmatrix(w).transpose()
    X_w_prod = np.matmul(X, w)
    error = (np.matmul(np.transpose(y - X_w_prod), y - X_w_prod)+lambd*np.matmul(np.transpose(w), w))/2
    error_grad = np.subtract(lambd*w,np.matmul(np.transpose(X), y-X_w_prod))
    # print("before array")
    # print(error_grad)
    
    # Reference: https://docs.scipy.org/doc/numpy-1.9.2/reference/generated/numpy.squeeze.html
    error_grad = np.squeeze(np.array(error_grad))

    # print("after array")
    # print(error_grad)                                       
    return error, error_grad

def mapNonLinear(x,p):
    # Inputs:                                                                  
    # x - a single column vector (N x 1)                                       
    # p - integer (>= 0)                                                       
    # Outputs:                                                                 
    # Xp - (N x (p+1)) 
    # IMPLEMENT THIS METHOD
    Xp = np.ones((x.shape[0], p + 1))
    for i in range(1, p + 1):
        Xp[:, i] = x ** i
    print("This is a check")
    if p == 0:
        print("Printing when p is 0")
        print(Xp)
    return Xp

# Main script

# Problem 1
# load the sample data                                                                 
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('sample.pickle','rb'),encoding = 'latin1')

# LDA
means,covmat = ldaLearn(X,y)
ldaacc,ldares = ldaTest(means,covmat,Xtest,ytest)
print('LDA Accuracy = '+str(ldaacc))
# QDA
means,covmats = qdaLearn(X,y)
qdaacc,qdares = qdaTest(means,covmats,Xtest,ytest)
print('QDA Accuracy = '+str(qdaacc))

# plotting boundaries
x1 = np.linspace(-5,20,100)
x2 = np.linspace(-5,20,100)
xx1,xx2 = np.meshgrid(x1,x2)
xx = np.zeros((x1.shape[0]*x2.shape[0],2))
xx[:,0] = xx1.ravel()
xx[:,1] = xx2.ravel()

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)

zacc,zldares = ldaTest(means,covmat,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zldares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
#plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('LDA')

plt.subplot(1, 2, 2)

zacc,zqdares = qdaTest(means,covmats,xx,np.zeros((xx.shape[0],1)))
plt.contourf(x1,x2,zqdares.reshape((x1.shape[0],x2.shape[0])),alpha=0.3)
plt.scatter(Xtest[:,0],Xtest[:,1],c=ytest.ravel())
plt.title('QDA')

plt.show()
# Problem 2
if sys.version_info.major == 2:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'))
else:
    X,y,Xtest,ytest = pickle.load(open('diabetes.pickle','rb'),encoding = 'latin1')

# add intercept
X_i = np.concatenate((np.ones((X.shape[0],1)), X), axis=1)
Xtest_i = np.concatenate((np.ones((Xtest.shape[0],1)), Xtest), axis=1)

w = learnOLERegression(X,y)
mle = testOLERegression(w,Xtest,ytest)
mle_tr=testOLERegression(w,X,y)
print('Test MSE without intercept::'+str(mle))
print('Train MSE without intercept::'+str(mle_tr))
w_i = learnOLERegression(X_i,y)
#calculating L1 norm of weights for linear regression
sum=0
for x in w_i:
    sum+=abs(x)

mle_i = testOLERegression(w_i,Xtest_i,ytest)
mle_i_tr = testOLERegression(w_i,X_i,y)
print('Test MSE with intercept::'+str(mle_i))
print('Train MSE with intercept::'+str(mle_i_tr))
print('L1 norm of weights for Linear Regression::'+str(sum))


# Problem 3
sum = 0
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses3_train = np.zeros((k,1))
mses3 = np.zeros((k,1))
for lambd in lambdas:
    w_l = learnRidgeRegression(X_i,y,lambd)
    mses3_train[i] = testOLERegression(w_l,X_i,y)
    mses3[i] = testOLERegression(w_l,Xtest_i,ytest)
    i = i + 1
#calculating L1 norm of weights for Ridge Regression
    #taking lamba=0.06
sum = 0
w_k = learnRidgeRegression(X_i,y,0.06)
for w in w_k:
    sum+=abs(w)
print('L1 norm of weights for Ridge Regression::'+str(sum))
#minimum test mse when lambda=0.06
mseTest=testOLERegression(w_k,Xtest_i,ytest)
print('Minimum Test MSE for Ridge::'+str(mseTest))
#minimum train mse when lambda=0
w_tr = learnRidgeRegression(X_i,y,0)
mseTrain=testOLERegression(w_tr,X_i,y)
print('Minimum Train MSE for Ridge::'+str(mseTrain))

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.subplot(1, 2, 2)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')

plt.show()
# Problem 4
k = 101
lambdas = np.linspace(0, 1, num=k)
i = 0
mses4_train = np.zeros((k,1))
mses4 = np.zeros((k,1))
opts = {'maxiter' : 20}    # Preferred value.                                                
w_init = np.ones((X_i.shape[1],1))
# min_mse_train = 999999
# lambd_train = 0
# min_mse_test = 999999
# lambd_test = 0
for lambd in lambdas:
    args = (X_i, y, lambd)
    w_l = minimize(regressionObjVal, w_init, jac=True, args=args,method='CG', options=opts)
    w_l = np.transpose(np.array(w_l.x))
    w_l = np.reshape(w_l,[len(w_l),1])
    mses4_train[i] = testOLERegression(w_l,X_i,y)
    mses4[i] = testOLERegression(w_l,Xtest_i,ytest)
#     if mses4_train[i][0] < min_mse_train:
#         min_mse_train = mses4_train[i][0]
#         lambd_train = lambd
#     if mses4[i][0] < min_mse_test:
#         min_mse_test = mses4[i][0]
#         lambd_test = lambd
    i = i + 1
fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(lambdas,mses4_train)
plt.plot(lambdas,mses3_train)
plt.title('MSE for Train Data')
plt.legend(['Using scipy.minimize','Direct minimization'])

plt.subplot(1, 2, 2)
plt.plot(lambdas,mses4)
plt.plot(lambdas,mses3)
plt.title('MSE for Test Data')
plt.legend(['Using scipy.minimize','Direct minimization'])
plt.show()
# print("min mse for training {}".format(min_mse_train))
# print("min mse for test {}".format(min_mse_test))
# print("min lambda for training {}".format(lambd_train))
# print("min lambda for test {}".format(lambd_test))




# Problem 5
pmax = 7
lambda_opt = 0.06 # REPLACE THIS WITH lambda_opt estimated from Problem 3
mses5_train = np.zeros((pmax,2))
mses5 = np.zeros((pmax,2))
for p in range(pmax):
    Xd = mapNonLinear(X[:,2],p)
    Xdtest = mapNonLinear(Xtest[:,2],p)
    w_d1 = learnRidgeRegression(Xd,y,0)
    mses5_train[p,0] = testOLERegression(w_d1,Xd,y)
    mses5[p,0] = testOLERegression(w_d1,Xdtest,ytest)
    w_d2 = learnRidgeRegression(Xd,y,lambda_opt)
    mses5_train[p,1] = testOLERegression(w_d2,Xd,y)
    mses5[p,1] = testOLERegression(w_d2,Xdtest,ytest)

fig = plt.figure(figsize=[12,6])
plt.subplot(1, 2, 1)
plt.plot(range(pmax),mses5_train)
print(mses5_train[6])
plt.title('MSE for Train Data')
plt.legend(('No Regularization','Regularization'))
plt.subplot(1, 2, 2)
plt.plot(range(pmax),mses5)
print(mses5[1])
plt.title('MSE for Test Data')
plt.legend(('No Regularization','Regularization'))
plt.show()





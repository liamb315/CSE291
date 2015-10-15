'''Logistic Regression for 0/1 in MNIST dataset'''
import numpy as np
import helper_functions as fn
import math_softmax as ms
from softmaxregression import SklearnSoftmaxRegression
from logisticregression import SklearnLogisticRegression
from sklearn import linear_model
import time
import matplotlib.pyplot as plt

#--------------------#
# Softmax Regression #
#--------------------#
# Load dataset from MNIST
full_trainarray = np.load('data/numpy/trainarray.npy')
full_trainlabel = np.load('data/numpy/trainlabel.npy')
full_testarray  = np.load('data/numpy/testarray.npy' )
full_testlabel  = np.load('data/numpy/testlabel.npy' )

X_train, Y_train = fn.preprocess_data(full_trainarray, full_trainlabel, False)
X_test, Y_test   = fn.preprocess_data(full_testarray, full_testlabel, False)

'''
# 0.  Sklearn softmax regression
print 'Softmax regression using sklearn'
t0 = time.time()
softmax = SklearnSoftmaxRegression(tolerance = 0.1)
W = softmax.train(X_train, Y_train)
t1 = time.time()
print ' Training time:', t1-t0
p = softmax.predict(X_test)
t2 = time.time()
print ' Testing time:', t2-t1
fn.print_performance(p, Y_test)
'''

# 0.  Softmax regression using stochastic gradient descent
print 'Softmax regression using stochastic gradient descent'
t0 = time.time()
W = np.zeros([X_train.shape[1], 10])
W = ms.stochastic_gradient_descent(X_train, Y_train, W, 10, 256	)
t1 = time.time()
print ' Training time:', t1-t0
p = ms.predict(X_test, W)
t2 = time.time()
print ' Testing time:', t2-t1
fn.print_performance(p, Y_test)
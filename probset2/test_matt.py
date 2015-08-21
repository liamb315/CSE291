'''Logistic Regression for 0/1 in MNIST dataset'''
import gradientdescent as gd
import numpy as np
import helper_functions as fn
import math_functions as mf
import logisticregression
import os

#Module is updated
reload(gd)

# Load datasety from MNIST
full_trainarray = np.load(os.path.join('data','numpy','trainarray.npy'))
full_trainlabel = np.load(os.path.join('data','numpy','trainlabel.npy'))
full_testarray  = np.load(os.path.join('data','numpy','testarray.npy' ))
full_testlabel  = np.load(os.path.join('data','numpy','testlabel.npy' ))
X_train, Y_train = fn.preprocess_data(full_trainarray, full_trainlabel)
X_test, Y_test   = fn.preprocess_data(full_testarray, full_testlabel)

'''
# Logistic Regression using sklearn
logreg = logisticregression.LogisticRegression(X_train, Y_train, X_test, Y_test)
logreg.fit()

print logreg.w1
predict = logreg.predict()
print logreg.incorrect

w = np.random.rand(X_train.shape[1])
w = mf.gradient_descent(X_train, Y_train, w, 1000)

print w
'''

#Testing gradientdescent.m
gdo=gd.GradientDescent(X_train, Y_train, X_test, Y_test)
gdo.batch_gd()
gdo.plot_convergence()

	



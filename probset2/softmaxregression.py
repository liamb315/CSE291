import numpy as np
import helper_functions as fn
from sklearn import linear_model


class SklearnSoftmaxRegression:
	'''sklearn based multiclass logistic (softmax) regression'''
	def __init__(self, tolerance):
		self.model     = linear_model.LogisticRegression(tol = tolerance)
		self.W         = None

	def train(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)
		self.W = self.model.coef_
		return self.W

	def predict(self, X_test):
		predict = self.model.predict(X_test)
		return predict		


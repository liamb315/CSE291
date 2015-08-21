import numpy as np
import os
import helper_functions as fn
from sklearn import linear_model


class SklearnLogisticRegression:
	'''sklearn based logistic/softmax regression wrapper'''
	def __init__(self):
		self.model     = linear_model.LogisticRegression()
		self.w         = None
		
	def train(self, X_train, Y_train):
		self.model.fit(X_train, Y_train)
		self.w = self.model.coef_
		return self.w

	def predict(self, X_test):
		predict = self.model.predict(X_test)
		return predict


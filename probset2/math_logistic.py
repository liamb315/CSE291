import numpy as np
import random, math
from scipy.misc import logsumexp


def sigmoid(x, w):
	'''Sigmoid function'''
	return 1.0/(1.0+np.exp(-1.0*np.dot(x, w)))	


def predict_logistic(X, w):
	'''Predict X using w'''
	pred = np.zeros(len(X))

	for i in range(0, len(X)):
		if sigmoid(X[i], w) <= 0.5:
			pred[i] = 0
		elif sigmoid(X[i], w) > 0.5:
			pred[i] = 1
	return pred


def loss_logistic(X, Y, w):
	'''Loss Function'''
	val = 0.0
	# Overflow issue solution:
	# http://lingpipe-blog.com/2012/02/16/howprevent-overflow-underflow-logistic-regression/
	for i in range(0, len(X)):
		val += -1.0 * (Y[i]*-1.0*logsumexp([0, -1.0*np.dot(X[i], w)]))
		val += -1.0 * ((1.0-Y[i])*-1.0*logsumexp([0, np.dot(X[i], w)]))
	return val


def gradient(X, Y, w):
	'''Gradient Loss Function calculated from all training examples
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
	
	Output:
		Gradient of loss function at w'''
	grad = np.zeros(len(w))

	for i in range(0, len(X)):
		grad += X[i]*(sigmoid(X[i], w)-Y[i])
	return grad


def gradient_batch(X, Y, w, n):
	'''Gradient Loss Function calculated from a batch of training examples
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
		3.  Batch size of number of examples to consider, n
	
	Output:
		Gradient of loss function at w
	'''
	grad = np.zeros(len(w))

	# Retrieve a random subset of samples
	indices = random.sample(xrange(len(X)), n)
	X_rand  = X[indices]
	Y_rand  = Y[indices]

	for i in range(0, len(X_rand)):
		grad += X[i]*(sigmoid(X_rand[i], w)-Y_rand[i])
	return grad


def gradient_descent(X, Y, w, M):
	'''Gradient descent of Loss Function using backtracking
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
		3.  Max Number of Iterations, M
	
	Output:
		Optimized Weight Vector,      w
	Further information:
	# Backtracking:  http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf	'''

	alpha = 0.15
	beta  = 0.5

	for i in range(0, M):
		eta = 1.0
		grad = gradient(X, Y, w)
		loss = loss_logistic(X, Y, w)
		print ' loss at iteration ',i,': ', loss
		while loss_logistic(X, Y, (w-eta*grad)) >= (loss - alpha*eta*np.linalg.norm(grad)):
			eta = beta * eta
			if eta < 10E-8:
				break
		w = w - eta * grad

	return w


def stochastic_gradient_descent(X, Y, w, M, n):
	'''Stochastic gradient descent of Loss Function using backtracking
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
		3.  Max Number of Iterations, M
		4.  Batch Size, n
	
	Output:
		Optimized Weight Vector,      w
	Further information:
	http://ufldl.stanford.edu/tutorial/supervised/OptimizationStochasticGradientDescent/ '''

	alpha = 0.15
	beta  = 0.5

	for i in range(0, M):
		eta = 0.1
		grad = gradient_batch(X, Y, w, n)
		loss = loss_logistic(X, Y, w)
		print ' loss at iteration ',i,': ', loss
		while loss_logistic(X, Y, (w-eta*grad)) >= (loss - alpha*eta*np.linalg.norm(grad)):
			eta = beta * eta
			if eta < 10E-8:
				break
		w = w - eta * grad

	return w


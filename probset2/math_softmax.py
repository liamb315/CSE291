import numpy as np
import random, math
import random
from scipy.misc import logsumexp


def softmax(W, x):
	'''Computes the softmax probability for a W-matrix'''
	prob  = np.dot(W, x)
	prob  = np.exp(prob)
	denom = np.sum(prob)
	return prob/denom

def predict(X, W):
	'''Predict X using W'''
	pred = np.zeros(len(X))

	for i in range(0, len(X)):
		pred[i] = np.argmax(softmax(W.T, X[i]))

	return pred


def identity(y, k):
	'''Identiy function that checks if integers match'''
	if y == k:
		return 1
	else:
		return 0


def loss_softmax(W, X, Y):
	'''Loss function for softmax regression'''
	val = 0.0
	for i in range(0, len(X)):
		for k in range(0, 10):
			vec  = np.dot(W.T, X[i]) 
			val -= identity(Y[i], k)*(np.dot(W[:,k], X[i]) - logsumexp(vec))
	return val


def gradient_batch(X, Y, W, n):
	'''Gradient Loss Function calculated from a batch of training examples
	Input:
		0.  Training Examples Matrix, X.
		1.  Training Labels Vector,   Y
		2.  Initalized Weight Vector, w
		3.  Batch size of number of examples to consider, n
	
	Output:
		Gradient of loss function at w
	'''
	val = np.zeros(W.shape)

	# Retrieve a random subset of samples
	indices = random.sample(xrange(len(X)), n)
	X_rand  = X[indices]
	Y_rand  = Y[indices]

	for i in range(0, len(X_rand)):
		for k in range(0, 10):
			val[:,k] -= X_rand[i]*(identity(Y_rand[i], k)-softmax(W.T, X_rand[i])[k])
	return val


def stochastic_gradient_descent(X, Y, W, M, n):
	'''Gradient descent of using backtracking
	Input:
		0.  Training Examples Matrix (m, , X.
		1.  Training Labels Vector (m, 1),   Y
		2.  Initalized Weight Matrix (n, k), W
		3.  Max Number of Iterations, M
		4.  Batch Size, n
	
	Output:
		Optimized Weight Matrix (n, k),      W
	Further information:
	# Backtracking:  http://users.ece.utexas.edu/~cmcaram/EE381V_2012F/Lecture_4_Scribe_Notes.final.pdf	'''

	'''
	alpha = 0.15
	beta  = 0.5

	for i in range(0, M):
		eta = 1E-1
		grad = gradient_batch(X, Y, W, n)
		loss = loss_softmax(W, X, Y)
		print ' loss at iteration ',i,': ', loss_softmax(W, X, Y)
		while loss_softmax((W-eta*grad), X, Y) >= (loss - alpha*eta*np.linalg.norm(grad)):
			eta = beta * eta
			if eta < 10E-5:
				break

		W = W - eta*grad
	''' 
	eta   = 1E-3
	decay = 0.9
	for i in range(0, M):
		grad = gradient_batch(X, Y, W, n)
		loss = loss_softmax(W, X, Y)
		print ' loss at iteration ',i,': ', loss_softmax(W, X, Y)
		W = W - eta*grad
		eta = eta/(1+i*decay)
	return W

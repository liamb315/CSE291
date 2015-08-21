import numpy as np
import os
import helper_functions as fn
import sys

#For logistic regression 
class GradientDescent:     
    def __init__(self, X_train, Y_train, X_test, Y_test):
		self.model     = None
		self.incorrect = []
		self.X_train   = X_train
		self.X_test    = X_test
		self.Y_train   = Y_train
		self.Y_test    = Y_test
		self.reset()

#sigmoid(x)
    def sigmoid(self, x):
        return 1/(1+np.exp(-x)) 
    
#J
    def loss_func(self, x, y, w):
        this_e=np.dot(w,np.transpose (x))
        return y.dot(np.divide(1, self.sigmoid(-this_e) + self.sigmoid(-(1-this_e))))
    
#log(J), e=x^tw
    def logliklihood(self, x, y, w):
        this_e=np.dot(w,np.transpose (x))
        return y.dot(np.divide(1, np.log10(self.sigmoid(-this_e)) + np.log10(self.sigmoid(-(1-this_e))))), this_e 
    
#Compute gradient and approximation of sigmoid 
    def sig_grad(self, x, y, w): 
        
        #Epsilon        
        ep=1e-4
        
        #(x^T)w    
        z=-np.dot(x, w) 
        
        #(x^T)(H(x)-y) 
        gradient = np.dot(np.transpose(x), np.log10(self.sigmoid(z)) - y ) 
        
        #Calculate gradient epsilon-approximation
        g1=self.loss_func(x, y, w+ep)
        g0=self.loss_func(x, y, w-ep)

        return gradient, np.divide(g1-g0, 2*ep)
    
#Batch gradient descent
    def batch_gd(self, maxiter=1000):
        self.reset()

        stepsz=np.array(1e-5)
        w=self.w
        this_log_lkhd=0

        print 'batch_gd: beginning'
        n=0
        while 1:
            (this_log_lkhd, e)=self.logliklihood(self.X_train, self.Y_train, w)
            last_gain=np.absolute(self.log_lkhd[-1] - np.array(this_log_lkhd))
    
            #Compute/Save logliklihood   
            self.log_lkhd[n]=this_log_lkhd

            #gradient
            (this_grad, this_grad_approx) = self.sig_grad(self.X_train, self.Y_train, w )

            #Compute gradient magnitudes
            gradnorm=np.linalg.norm(this_grad)
            approx_gradnorm=np.linalg.norm(this_grad_approx)            
                        
            print 'iteration: liklihood: liklihood difference | gradient: gradient check'
            print('%d:  %f:  %f | %f:  %f'% (n, this_log_lkhd, last_gain,gradnorm, approx_gradnorm ))
            sys.stdout.flush()

            #Compute/save logliklihood difference
            last_gain=np.absolute(self.log_lkhd[n-1]-this_log_lkhd)
            
            #Convergence condition
            if last_gain < 1e-9: break 
        
            #Compute gradient
            self.gradient.append(self.sig_grad(self.X_train, self.Y_train, w))

            #update           
            w += stepsz*this_grad
            n+=1
    
#Stochastic gradient descent
    def stoch_gd(self):      
        self.reset()
        w = np.random.rand(X_train.shape[1])
        pass

#plot convergence results
    def plot_convergence(self):
        pass

#reset before initiating new method 
    def reset(self): 
        self.log_lkhd=np.array([0],dtype=np.float_)
        self.stepsz=np.array([0],dtype=np.float_)
        self.params=np.array([0],dtype=np.float_)
        self.w=self.X_train[:, np.random.randint(1, self.X_train.shape[2], 1)]
        self.gradient=[]


if __name__=='__main__':
    # Load dataset from MNIST
    full_trainarray = np.load(os.path.join('data','numpy','trainarray.npy'))
    full_trainlabel = np.load(os.path.join('data','numpy','trainlabel.npy'))
    full_testarray  = np.load(os.path.join('data','numpy','testarray.npy' ))
    full_testlabel  = np.load(os.path.join('data','numpy','testlabel.npy' ))
    
    
    X_train, Y_train = fn.preprocess_data(full_trainarray, full_trainlabel)
    X_test, Y_test = fn.preprocess_data(full_testarray, full_testlabel)
    gdo=GradientDescent(X_train, Y_train, X_test, Y_test)
    gdo.batch_gd()
    gdo.plot_convergence()

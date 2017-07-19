''' Linear regression solving with gradient descent'''
''' suitable for large number of features'''
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class my_BatchRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, alpha = .1, max_iter = 1000,fit_intercept = True,l2 = 0,\
                 initial_theta = None, verbose = 0, random_state = 42, history = None):
        # learning rate
        self.alpha = alpha
        # maximum iteration which determines stopping condition
        self.max_iter = max_iter
        # regularization parameter
        self.l2 = l2
        # initial weights
        self.initial_theta = initial_theta
        # wether to add intercept(a column of 1)
        self.fit_intercept = fit_intercept
        # simple verbosity level(just 0,1)
        self.verbose = verbose
        # random state: here we use random sample to genetare initial weights
        self.random_state = random_state
        #for saving convergence preprocess
        self.history = history
        # *** for gradient descent, the train data should be normalized
        # which is done in the "prepare data for model" phase
    def fit(self, X, y):
        assert (len(X) == len(y)), " X and y must have same length"
        # number of data points
        X_copy = np.array(X)
        N = X.shape[0]
        if self.fit_intercept == True:
            # add bias term
            X = np.c_[np.ones((N,1)), X]
        # number of features
        d = X.shape[1]
        np.random.seed(self.random_state)
        if self.initial_theta is None:
            self.initial_theta = np.random.randn(d,1)
        theta = np.array(self.initial_theta)

        for iteration in range(self.max_iter):
            # batch gradient:in each iteration,gradient is computed over all dpts
            a = (self.l2/float(N))*theta
            a[0,0] =0
            gradients = (2./N)*X.T.dot(X.dot(theta) - y) + a


            # update theta
            theta = theta - self.alpha*gradients
            # if a list name is given in fit method, save obj in each iteration
            if self.history is not None:
                self.theta_best_ = theta
                self.history.append(self.rmse(X_copy,y)[0,0])
            # print result every 10 iteration if verbose = 1
            if self.verbose:
                if iteration%10 == 0:
                    print "-"*10 + "iteration"+str(iteration) + "-"*10
                    print self.rmse(X_copy,y)[0,0]
        self.theta_best_ = theta
        return self

    def predict(self, X, y= None):
        if self.fit_intercept == True:
            N = X.shape[0]
            X = np.c_[np.ones((N,1)), X]
        return X.dot(self.theta_best_)

    def rmse(self, X, y):
        prediction = self.predict(X)
        return np.sqrt( (prediction - y).T.dot(prediction - y)/(float(len(X))) )

    def score(self, X, y):
        prediction = self.predict(X)
        return -(prediction - y).T.dot(prediction - y)/float(len(X))

# import os
# os.system('cls')
# np.random.seed(42)
# X = 2*np.random.rand(100,1)
# y = 4 + 3* X + np.random.randn(100,1)
# # observing the effect of alpha
# linreg = my_BatchRegressor(alpha = .5)
# a = []
# linreg.fit(X,y, history = a)
# import matplotlib.pyplot as plt
# plt.plot(range(10), a[:10])
# plt.show()

''' simple linear regression (normal equation)'''

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

class my_LinearRegression(BaseEstimator, RegressorMixin):
    '''the same as sklearn.linear_model.LinearRegression'''

    def __init__(self, fit_intercept = True, l2 = 0):
        self.fit_intercept = fit_intercept
        self.l2 = l2

    def fit(self, X, y):
        assert (len(X) == len(y)), " X and y must have same length"
        X_copy = np.array(X)
        # number of data points
        N = X.shape[0]
        if self.fit_intercept == True:
            # adding intercept
            X = np.c_[np.ones((N,1)), X]
        # number of features
        d = X.shape[1]
        # normal equation
        A = np.eye(d)
        A[0,0] = 0
        self.theta_best_ =\
         np.linalg.pinv((X.T).dot(X) + (self.l2*A)).dot(X.T).dot(y)
        return self

    def predict(self, X, y = None):
        X_copy = np.array(X)
        # number of data
        N = X.shape[0]
        if self.fit_intercept == True:
            # add bias term
            X = np.c_[np.ones((N,1)), X]
        return X.dot(self.theta_best_)

    def score(self, X, y):
        # it return the negative mean square error
        # the reason for negative: for gridsearch the bigger score the better
        # although you can pass 'neg_mean_squared_error' as scoring metric
        prediction = self.predict(X)
        return  -( (prediction - y).T.dot(prediction - y) /float(len(X)) )
    def rmse(self, X,y):
        prediction = self.predict(X)
        return  np.sqrt( (prediction - y).T.dot(prediction - y) /float(len(X)) )


# import os
# os.system('cls')
# np.random.seed(42)
# X = 2*np.random.rand(100,1)
# y = 4 + 3* X[:98] + np.random.randn(97)
#
# linreg = my_LinearRegression()
# linreg.fit(X,y)
# score = linreg.score(X,y)
# print np.sqrt(-score)
# print linreg.theta_best_
# X_new = np.array([[0], [2]])

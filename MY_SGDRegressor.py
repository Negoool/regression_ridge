from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np
from sklearn.utils import check_X_y
import os
os.system('cls')
class my_SGDRegressor(BaseEstimator, RegressorMixin):
    ''' like 'sklearn.linear_model.SGDRegressor but different learning rate'''
    ''' methods : __init__ , fit, predict, score, rmse
        attributes : theta_best_'''

    def __init__(self, alpha0 = .1, max_epoch = 50,fit_intercept = True, l2 = 0,\
     initial_theta = None, random_state = 42, learning_schedule = 'invT', history = None ):

        # initial learning rate
        self.alpha0 = alpha0
        # maximim number of epoch
        self.max_epoch = max_epoch
        # boolean: wether or not add bias term
        self.fit_intercept = fit_intercept
        # regularization parameter ridge
        self.l2 = l2
        # initial weights
        self.initial_theta = initial_theta
        # random state
        self.random_state = random_state
        # learning schedule: 2 states: Constant or invT
        self.learning_schedule = learning_schedule
        # wether or not save J for observing convergence
        self.history = history

    def fit(self, X, y):
        # check length of given X and y
        if len(X) != len(y):
            raise ValueError('shape of X and y should be tha same')
        X_copy = np.array(X)
        # number of data points
        N = X.shape[0]
        if self.fit_intercept == True: # add bias
            X = np.c_[np.ones((N,1)), X]

        np.random.seed(self.random_state)
        # number of features(inclusing intercept)
        d = X.shape[1]
        if self.initial_theta is None:
            self.initial_theta = np.random.randn(d,1)
        theta = self.initial_theta

        for epoch in range(self.max_epoch):
            for i in range(N):
                j = np.random.randint(N)
                # shape of X[j:j+1] is 1*d, but X[j,:] is not (dL, )
                gradients = (2*(X[j:j+1].dot(theta) - y[j:j+1])*X[j:j+1].T) + theta*self.l2
                if self.learning_schedule == 'invT':
                    # alpha = (t0/t+t1)
                    # there are many other learning schedules
                    t1 = 50
                    t0 = t1 * self.alpha0
                    t = (epoch*N)+i
                    alpha = np.float(t0)/(t1+t)
                if self.learning_schedule == 'Constant':
                    alpha = self.alpha0
                theta = theta - alpha * gradients

                if self.history is not None:
                    self.theta_best_ = theta # so other methods can access that
                    # in  predict method it is ganna add bias again, so pass the
                    # version without bias
                    self.history.append(self.rmse(X_copy,y)[0,0])
        self.theta_best_ = theta
        return self

    def predict(self, X, y = None):
        if self.fit_intercept == True:
            N = X.shape[0]
            X = np.c_[np.ones((N,1)), X]
        return X.dot(self.theta_best_)

    def rmse(self, X,y):
        # return rEsidua Sum of square
        prediction = self.predict(X)
        return np.sqrt( (prediction - y).T.dot(prediction - y)/float(len(X)) )

    def score(self, X,y):
        prediction = self.predict(X)
        # introduced so gridsearch, the higher the better
        # same as 'neg_mean_squared_error'
        return  -(prediction - y).T.dot(prediction - y)/float(len(X))

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from plot_curves import plot_learning_curves
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
os.system('cls')

np.random.seed(42)
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
# X_new = np.linspace(0, 3, 100).reshape(100, 1)
his = []
## transformation(adding polynomial features) and model in pipeline format

Pipeline_poly_reg = Pipeline([\
  ('poly', PolynomialFeatures(degree=10,  include_bias=False) ),
  ('scaler', StandardScaler(with_mean=False)),
  ('linreg', my_SGDRegressor(l2 = .1, alpha0 = .04,max_epoch = 500, history = his))\
  ])
Pipeline_poly_reg.fit(X, y)

# the first 1 refers to the second element of pipeline which is a tuple
# ( name, estimator) and the second 1 refers to the estimator not its name
print "costant:", Pipeline_poly_reg.steps[2][1].theta_best_[0]
print "real weights:\n", Pipeline_poly_reg.steps[2][1].theta_best_[1:]\
 .ravel()/Pipeline_poly_reg.steps[1][1].scale_

print "rmse on train set",np.sqrt(-1*Pipeline_poly_reg.score(X,y))
#polt learning curve
# plot_learning_curves(Pipeline_poly_reg, X,y)
# plt.figure()
X_new = np.linspace(0, 3, 100).reshape(100, 1)
y_new = Pipeline_poly_reg.predict(X_new)
plt.plot(X_new, y_new)

plt.figure()
plt.plot(range(10000), his)
plt.show()

''' linear regression for predicting house prices'''

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import copy

import os
os.system('cls')

# import data
data = pd.read_csv('kc_house_train_data.csv')
data_test = pd.read_csv('kc_house_test_data.csv')
data.info()
#print data.describe()

## visualization
# data.plot(kind= 'box', sharex= False, subplots = True, layout = (4,5))
#
# data.hist()
#
# data.plot(x = 'bedrooms', y = 'price', kind ='scatter')
#
# data.plot(x = 'lat', y = 'price', kind ='scatter')
#
# data.plot(x = 'long', y = 'price', kind ='scatter', color = 'k')
#
# plt.figure()
# data['sqft_living'].plot(kind = 'box')

## withdraw outliers

## define new features

## convert dataframe to numpy array
def convert_to_np(dataframe, features_list, target):
    ''' converert data(dataframe format) to feature and output(np array)'''
    dataframe['constant'] = 1
    features_list = ['constant'] + features_list
    x = dataframe.as_matrix(features_list)
    y = dataframe.as_matrix( target)

    return x,y

my_features = ['bedrooms','bathrooms','sqft_living','sqft_lot','floors',
                'waterfront', 'view', 'condition', 'grade','sqft_above',
                'sqft_basement','yr_built','yr_renovated']

x, y = convert_to_np(data, my_features, ['price'])
x_test, y_test = convert_to_np(data_test, my_features, ['price'])


def scale(x):
    ''' scale features(x)
    here we devide all atributes by their coresponding maxixmum
    so all features are in the range of (0,1)
    for using gradient descent
    '''
    xmax = np.amax(x, axis=0)
    x_scaled = x/np.array(xmax, dtype = np.float)
    return x_scaled

x_scaled = scale(x)

def split(data, percent):
    ''' split data points to train and validation ( later for choosing landa)'''
    # random_indices = np.random.choice(data.shape[0], size = data.shape[0]/percent, replace = False )
    # data1 = data[random_indices]
    # data2 = np.delete(data, random_indices, axis=0)
    copy_data = copy.copy(data)
    np.random.shuffle(copy_data)
    n = np.round(data.shape[0]*percent)
    data1 = copy_data[:n , :]
    data2 = copy_data[n: , :]
    return data1, data2

x_train, x_valid = split(x, .9)
y_train , y_valid = split(y, .9)

## implement gradient descent or close form on train data
##1 close form
def RSS(x, y , teta):
    N = x.shape[0]
    y_predicted = np.dot(x, teta)
    rss = (1./(2*N))*np.dot((y - y_predicted).T, (y - y_predicted))
    return rss[0]

def regression_close_form(x, y, lamb = 0):
    dimension = x.shape[1]
    I = np.eye(dimension)
    I[0,0 ] = 0
    teta = np.dot(np.linalg.pinv(np.dot(x.T, x) + lamb*I) , np.dot(x.T, y))
    return teta

# basic results
show_result1 = 0
if show_result1 == 1:
    teta_scaled = regression_close_form(x_scaled, y)
    print "RSS_TRAIN : ", RSS(x_scaled, y , teta_scaled)

    xmax = np.amax(x, axis=0)
    teta = teta_scaled/np.array(xmax[:, np.newaxis], dtype = np.float)
    print teta

    print "rss_train :", RSS(x, y , teta)
    print "rss_test :", RSS(x_test, y_test , teta)

rss_train = []
rss_test = []
for lamb in range(0,10000,1000):
    teta = regression_close_form(x_train, y_train, lamb)

    rss_train.append(RSS(x_train, y_train , teta))
    rss_test.append(RSS(x_valid, y_valid , teta))


plt.plot( range(0,10000,1000), rss_train, 'bo', label = 'rss for train')
plt.plot( range(0,10000,1000), rss_test, 'ko', label = 'rss for test')
plt.legend()
plt.show()


## gradient descent
def regression_gradient_descent(x_scaled, y, learning_rate, max_iter = 300, initial_weights = None):

    num_data = x_scaled.shape[0]
    dimension = x_scaled.shape[1]

    if initial_weights is None:
        initial_weights = np.zeros((dimension,1))

    weights = initial_weights[:]
    print weights.shape
    j = []

    for iter in range(max_iter):

        prediction = np.dot(x_scaled, weights)
        derivitive = (1./num_data) * np.dot(x_scaled.T, (prediction - y))

        weights = weights - (learning_rate * derivitive)

        rss = RSS(x_scaled, y , weights)

        j.append(rss)
    print rss


    return weights, j


# a, b= regression_gradient_descent(x_scaled, y, 6.2e-1, max_iter = 100000)
# print a
# plt.plot(b)
# plt.show()



## use the result for test data

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
data_train_total = pd.read_csv('kc_house_train_data.csv')
data_test_total = pd.read_csv('kc_house_test_data.csv')

#data.info()
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

# to observe overfitting and regularization we may increase # features or
# reduce number of data points
data = data_train_total.sample(80, axis = 0)
data_test = data_test_total.sample(50,axis = 0)

## withdraw outliers

## define new features
data['sqrt_sqft_living'] = np.sqrt(data['sqft_living'])
data['sqrt_sqft_lot'] = np.sqrt(data['sqft_lot'])
data['square_bed'] = data['bedrooms']* data['bedrooms']
data['bedbath'] = data['bedrooms']* data['bathrooms']

data_test['sqrt_sqft_living'] = np.sqrt(data_test['sqft_living'])
data_test['sqrt_sqft_lot'] = np.sqrt(data_test['sqft_lot'])
data_test['square_bed'] = data_test['bedrooms']* data_test['bedrooms']
data_test['bedbath'] = data_test['bedrooms']* data_test['bathrooms']

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
                'sqft_basement','yr_built','yr_renovated', 'sqrt_sqft_living',
                'sqrt_sqft_lot', 'square_bed', 'bedbath' ]

x, y = convert_to_np(data, my_features, ['price'])
x_test, y_test = convert_to_np(data_test, my_features, ['price'])


# def split(x, y, percent):
#     ''' split data points to train and validation ( later for choosing landa)'''
#     random_indices = np.random.choice(x.shape[0], size = np.round(x.shape[0]*percent), replace = False )
#     x1 = x[random_indices]
#     x2 = np.delete(x, random_indices, axis=0)
#
#     y1 =y[random_indices]
#     y2 = np.delete(y, random_indices, axis=0)
#     return x1,y1,x2,y2
#     # copy_x = copy.copy(x)
#     # copy_y = copy.copy(y)
#     # np.random.shuffle(copy_data)
#     # n = np.round(data.shape[0]*percent)
#     # data1 = copy_data[:n , :]
#     # data2 = copy_data[n: , :]
#     #return data1, data2

def RSS(x, y , teta):
    ''' compute Risidual Sum of Suare for the given weights'''
    N = x.shape[0]
    y_predicted = np.dot(x, teta)
    rss = (1./(2*N))*np.dot((y - y_predicted).T, (y - y_predicted))
    return rss[0]

##1 close form
def regression_close_form(x, y, lamb = 0):
    ''' L2 regularized linear regression
    parameter "lamb" controls the bias/tradoff balances
    '''
    dimension = x.shape[1]
    I = np.eye(dimension)
    # intercept is not regularized
    I[0,0 ] = 0
    teta = np.dot(np.linalg.pinv(np.dot(x.T, x) + lamb*I) , np.dot(x.T, y))
    return teta

weights = regression_close_form(x, y, lamb = 0)
prediction = np.dot(x,weights)
plt.plot(prediction, 'ro')
plt.plot(y , 'bo')
plt.title(' real price and prediction for train data for lambda = 0')



col = np.array(range(17))
rss_train = []
rss_test = []
lamb_range = range(0,1000,100)

plt.figure()
plt.title( " weights of features as lambda increases")
plt.xlabel('lambda')
plt.ylabel('all weights except constant')
for lamb in lamb_range:
    teta = regression_close_form(x, y, lamb)
    plt.scatter(np.array([lamb]*17), teta[1:], c = col)
    rss_train.append(RSS(x, y , teta))
    rss_test.append(RSS(x_test, y_test , teta))
#
plt.figure()
plt.plot( lamb_range, rss_train, 'bo', label = 'RSS for train')
# this is just for observance but this is not used for parameter selection and training
plt.plot( lamb_range, rss_test, 'ko', label = 'RSS for test')
plt.legend()

def cross_validation(x, y, lamb_range, num_sections = 5):
    '''
    applying cross validation for choosing the best value for lambda
    which loose to the least average of rss on validation sets
    output : list of averge rss on validation set for different valuse of lamdbas
    '''
    num_data = x.shape[0]
    num_data_fold = num_data/5
    rss= []
    for lamb in lamb_range:
        rss_lambda = 0
        for k in range(num_sections):
            if k !=(num_sections-1):
                x_valid = x[(k*num_data_fold) : (k+1)*num_data_fold]
                y_valid = y[(k*num_data_fold) : (k+1)*num_data_fold]
                x_train  = np.delete(x, range( k*num_data_fold , (k+1)*num_data_fold), axis=0)
                y_train  = np.delete(y, range(k*num_data_fold , (k+1)*num_data_fold), axis=0)
            else :
                x_valid = x[(k*num_data_fold) :,: ]
                y_valid = y[(k*num_data_fold) :,: ]
                x_train = x[: (k*num_data_fold) ,: ]
                y_train = y[: (k*num_data_fold) ,: ]

            teta = regression_close_form(x_train, y_train, lamb)
            rss_lambda += RSS(x_valid, y_valid , teta)
        rss.append(rss_lambda/float(num_sections))
    return rss

rss = cross_validation(x, y, lamb_range)
plt.figure()
plt.plot(lamb_range, rss)
plt.xlabel('lambda')
plt.title(' average rss on the validation set for different lambda values')



plt.show()

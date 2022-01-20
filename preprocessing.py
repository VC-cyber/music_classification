import numpy as np
import pandas as pd

def preprocessing():
    data = pd.read_csv('features_3_sec 2.csv')

    cat = {'blues':1, 'classical':2,
    'country':3,'disco':4,'hiphop':5, 'jazz':6, 'metal':7, 'pop':8, 'reggae':9,'rock':10}

    data = np.array(data.iloc[0:-1])
    np.random.shuffle(data)
    x_train = data[1:7992,1:-1]
    y_train = data[1:7992, -1]

    for i in range(len(y_train)):
        y_train[i] = cat[y_train[i]] 
        
    x_test = data[7992:, 1:-1]
    y_test = data[7992:, -1]

    for i in range(len(y_test)):
            y_test[i] = cat[y_test[i]]

    x_train = x_train.astype(float)
    y_train = y_train.astype(int)
    x_test = x_test.astype(float)
    y_test = y_test.astype(int)

    return x_train, y_train, x_test, y_test

def preprocessingCV():
    data = pd.read_csv('features_3_sec 2.csv')

    cat = {'blues':1, 'classical':2,
    'country':3,'disco':4,'hiphop':5, 'jazz':6, 'metal':7, 'pop':8, 'reggae':9,'rock':10}

    data = np.array(data.iloc[0:-1])
    np.random.shuffle(data)
    x_train = data[1:6993,1:-1]
    y_train = data[1:6993, -1]

    for i in range(len(y_train)):
        y_train[i] = cat[y_train[i]] 
        
    x_cv = data[6993:8491, 1:-1]
    y_cv = data[6993:8491, -1]
    x_test = data[8491:, 1:-1]
    y_test = data[8491:, -1]

    for i in range(len(y_cv)):
            y_cv[i] = cat[y_cv[i]]

    for i in range(len(y_test)):
            y_test[i] = cat[y_test[i]]

    x_train = x_train.astype(float)
    y_train = y_train.astype(int)
    x_test = x_test.astype(float)
    y_test = y_test.astype(int)
    x_cv = x_cv.astype(float)
    y_cv = y_cv.astype(int)

    return x_train, y_train, x_test, y_test, x_cv, y_cv



"""
Author: michealowen
Last edited: 2019.11.14,Thursday
计算距离
"""
#encoding=UTF-8

import numpy as np

def euclidean_distance(x_1,x_2):
    '''
    计算x_1和x_2中的向量的欧几里得(L2)距离
    compute the L2-distances of the vectors in x_1,x_2
    
    Params:
        x_1:array-like, shape: (n_samples_X, n_features)
        x_2:array-like, shape: (n_samples_X, n_features)
    Returns:
        distance:array-like, shape: (n_samples_X,1)
    '''
    x_1 = check_array(x_1)
    x_2 = check_array(x_2)
    return np.sum(np.power(x_1-x_2,2),axis=1).reshape(-1,1)


def manhattan_distance(x_1,x_2):
    '''
    计算x_1和x_2中的向量的曼哈顿距离(L1)距离
    compute the L1-distances of the vectors in x_1,x_2

    Params:
        x_1:array-like, shape: (n_samples_X, n_features)
        x_2:array-like, shape: (n_samples_X, n_features)
    Returns:
        distance:array-like, shape: (n_samples_X,1)
    '''
    x_1 = check_array(x_1)
    x_2 = check_array(x_2)
    return np.sum(np.abs(x_1-x_2,2),axis=1).reshape(-1,1)

def check_array(x):
    '''
    to change the type of x to np.ndarray if isinstance(x,np.ndarray) != True
    ''' 
    if not isinstance(x,np.ndarray):
        x = np.array(x)
    return x


if __name__ == '__main__':
    print(manhattan_distance([1,1],[2,2]))

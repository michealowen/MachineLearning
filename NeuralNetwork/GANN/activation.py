"""
Author: michealowen
Last edited: 2019.11.29,Friday
激活函数模块
"""
#encoding=UTF-8

import numpy as np

def sigmoid(Z):
    #sigmoid函数
    return 1.0/(1.0+np.exp(-Z))
    

def new_sigmoid(z):
    '''
    改进版的sigmoid函数,避免溢出,分两种情况,输入参数为数字和输入参数为向量
    '''
    if isinstance(z, (int, float)):
        if z >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
            z = 1.0 / (1 + np.exp(-z))
        else:
            z = np.exp(z) / (1 + np.exp(z))
    else:
        for row_index, row_value in enumerate(z):
            for column_index , column_value in enumerate(row_value):
                if column_value >= 0:  # 对sigmoid函数的优化，避免了出现极大的数据溢出
                    z[row_index,column_index] = 1.0 / (1 + np.exp(-column_value))
                else:
                    z[row_index,column_index] = np.exp(column_value) / (1 + np.exp(column_value))        
    return z
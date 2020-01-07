"""
Author: michealowen
Last edited: 2019.11.29,Friday
归一化函数模块
"""
#encoding=UTF-8

import numpy as np

def Normalize(x):
    '''
    归一化
    '''
    #for i in range(x.shape[1]):
    #    x[:,i] = x[:,i]/51
    return x/51

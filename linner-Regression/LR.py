"""
Author: michealowen
Last edited: 2019.8.7,Wednsday
线性回归算法,使用波士顿房价数据集
"""
#encoding=UTF-8

import numpy as np
import pandas as pd
import sympy as sp
from sklearn import datasets

def loadData():
    """
    加载数据
    """
    data  = datasets.load_boston().data
    lable = datasets.load_boston().target
    return data,lable

def standardNormalize(X,Y):
    """
    使用正态分布标准化
    """
    #先对X归一化
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])
    #再对Y归一化
    Y = (Y - np.mean(Y))/np.std(Y)

    return X,Y

def normalize(X,Y):
    """
    0,1归一化
    """
    #先对X进行归一化
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.min(X[:,i]))/(np.max(X[:,i]) - np.min(X[:,i]))
    #再对Y归一化
    Y = (Y - np.min(Y))/(np.max(Y) - np.min(Y))
    #print(X)
    return X,Y

def GD(data,lable):
    """
    梯度下降,当迭代一万次时结束

    Args:
        data:参数
        lable:房价
    """
    #m为样本数,n为数据维度,p为迭代时候的步长
    m,n = data.shape
    w = np.array([0 for i in range(n+1)],dtype='float64')
    p = 0.1
    #给X扩充一列常数项
    X = np.c_[data,np.array([1 for i in range(m)])]
    Y = lable

    i = 0
    MinCost = float("inf")
    while i<10000000:
        J = 1/m * np.dot( (np.dot(X,w.T) - Y),(np.dot(X,w.T) - Y).T)
        if J < MinCost:
            MinCost = J
            print(J," ",i)
            w -= 2/m * p * (np.dot(X.T ,(np.dot( X ,w.T) - Y.T ))).T 
            i += 1 
        else:
            break
        
    return None 

def MBGD(data,lable):
    """
    部分批量梯度下降

    Args:
        data:参数
        lable:房价
    """
    #m为样本数,n为数据维度,p为迭代时候的步长
    m,n = data.shape
    w = np.array([0 for i in range(n+1)],dtype='float64')
    p = 0.1
    #给X扩充一列常数项
    X = np.c_[data,np.array([1 for i in range(m)])]
    Y = lable

    i = 0
    MinCost = float("inf")
    while True:
        partIndex = np.random.choice(range(m),int(m/10))
        X_part = X[partIndex]
        Y_part = Y[partIndex]
        J = 1/m * np.dot( (np.dot(X_part,w) - Y_part),(np.dot(X_part,w) - Y_part).T)
        if abs(J - MinCost) < 0.0001:
            break
        else:
            print("J:",J," ",i)
            w -= 2/m * p * (np.dot(X_part.T ,(np.dot( X_part ,w.T) - Y_part.T ))).T 
            i = i+1
            MinCost = J
    return None

def SGD(data,lable):
    """
    随机梯度下降

    Args:
        data:参数
        lable:房价
    """
    #m为样本数,n为数据维度,p为迭代时候的步长
    m,n = data.shape
    w = np.array([0 for i in range(n+1)],dtype='float64')
    p = 0.1
    #给X扩充一列常数项
    X = np.c_[data,np.array([1 for i in range(m)])]
    Y = lable

    i = 0
    MinCost = float("inf")
    while True:
        partIndex = np.random.randint(len(X))
        X_part = X[partIndex]
        Y_part = Y[partIndex]
        J = 1/m * np.dot( (np.dot(X_part,w) - Y_part),(np.dot(X_part,w) - Y_part).T)
        if abs(J - MinCost) < 0.0001:
            break
        else:
            print("J:",J," ",i)
            w -= 2/m * p * (np.dot(X_part.T ,(np.dot( X_part ,w.T) - Y_part.T ))).T 
            i = i+1
            MinCost = J
    return None


def MT(data,lable):
    """
    基于矩阵求解的方法
    """

    #m为样本数,n为数据维度,p为迭代时候的步长
    m = len(data)
    #w = np.array([0 for i in range(n+1)],dtype='float64')
    #给X扩充一列常数项
    X = np.c_[data,np.array([1 for i in range(len(data))])]
    Y = lable
    w = np.dot(np.linalg.pinv(X),Y.T).T
    J = 1/m * np.dot( (np.dot(X,w.T) - Y),(np.dot(X,w.T) - Y).T)
    #print(J)
    return None


data,lable = loadData()
#data,lable = standardNormalize(data,lable)
data,lable = normalize(data,lable)
#GD(data,lable)
MBGD(data,lable)
SGD(data,lable)
MT(data,lable)

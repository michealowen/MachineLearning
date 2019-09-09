"""
Author: michealowen
Last edited: 2019.8.9,Friday
Logistic回归(分类),使用Iris数据集,进行二分类
"""
#encoding=UTF-8

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData():
    """
    加载数据
    """
    data = datasets.load_iris().data
    lable = datasets.load_iris().target
    #从数据中取出两类
    data = np.c_[data,lable]
    #分为训练数据和测试数据
    trainData = np.array([data[i] for i in range(len(data)) if data[i][4]!=2 and i%5!=0])
    testData  = np.array([data[i] for i in range(len(data)) if data[i][4]!=2 and i%5==0])
    return trainData[:,[0,1,2,3]],trainData[:,4],testData[:,[0,1,2,3]],testData[:,4]

def standardNormalize(X):
    """
    使用正态分布标准化
    """
    #先对X归一化
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.mean(X[:,i]))/np.std(X[:,i])

    return X

def normalize(X):
    """
    0,1归一化
    """
    #先对X进行归一化
    for i in range(X.shape[1]):
        X[:,i] = (X[:,i] - np.min(X[:,i]))/(np.max(X[:,i]) - np.min(X[:,i]))
    print(X)
    return X

def GD(data,lable):
    """
    梯度下降法
    """

    m,n = data.shape
    #X加上常数列
    X = np.c_[data,[1 for i in range(m)]]
    Y = lable
    w = np.array([0 for i in range(n+1)],dtype='float64')

    i = 0
    p = 0.1
    MinCost = float("inf")
    while i < 10000:
        J = -1/m * (np.dot( np.dot(w,X.T),Y.T) + np.sum(np.log(np.exp(-1*np.dot(w,X.T))/(1+np.exp(-1*np.dot(w,X.T))))))
        if np.abs(J) < MinCost:
            MinCost = np.abs(J)
            print(J," ",i)
            w -= p* 1/m * np.dot( 1/(1+np.exp(-1*np.dot(w,X.T)))- Y ,X)
            i += 1
        else:
            break
    print(w)
    return w

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
    print(J)
    return w

def show(X,Y,w):
    """
    将逻辑回归的结果进行展示,特定为鸢尾花数据集(降维为3维)
    """
    for i in range(len(X)):
        if Y[i] == 0:
            plt.plot(np.dot(w[0:4],X[i])+w[4],1/(1+np.exp(-1*np.dot(w[0:4],X[i])-w[4])),'ob')
        else:
            plt.plot(np.dot(w[0:4],X[i])+w[4],1/(1+np.exp(-1*np.dot(w[0:4],X[i])-w[4])),'or')
    r = np.linspace(-10,10,100)
    t = 1/(1+np.exp(-r))
    plt.plot(r,t)
    plt.show()
    return None

'''
data,lable = loadData()
data,lable = normalize(data,lable)
GD(data,lable)
MT(data,lable)
'''

trainData,trainLable,testData,testLable = loadData()
trainData = normalize(trainData)
testData  = normalize(testData)
#使用训练数据训练
w = GD(trainData,trainLable)
#w = MT(trainData,trainLable)
#使用测试数据测试
show(testData,testLable,w)
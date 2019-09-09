"""
Author: michealowen
Last edited: 2019.7.16,Tuesday
实现朴素贝叶斯算法
"""
#coding=UTF-8

import numpy as np
import pandas as pd

def loadData(fileName):
    """
    加载数据
    通过pandas读取csv文件为DataFrame
    再将Dataframe转化为矩阵

    Args:
        fileName:文件名

    Returns:
        data:数据矩阵
    """
    data = pd.read_csv('data.csv')
    data = data.values
    return data

def bayes(data,t):
    """
    朴素贝叶斯分类

    Args:
        data:传入的数据
        t:进行预测的样本
    
    Returns:
        预测的结果
    """
    if(len(data)) > 0:
        dimension = len(data[0])
    match = {}
    resultType = {}
    for i in range(len(data)):
        if resultType.get(data[i][dimension-1]) == None:
            resultType[data[i][dimension-1]] = 0
        for j in range(dimension-1):
            #首先看该键是否存在
            if match.get((data[i][j],data[i][dimension-1])) == None:
                match[(data[i][j],data[i][dimension-1])] = 1
            else:
                match[(data[i][j],data[i][dimension-1])] +=1
    
  
    #y的取值空间,和对应的可能性,如P(y=1)=xxx,P(y=-1)=xxx
    for kk in match.keys():
        for jj in resultType.keys():
            if kk[1] == jj:
                resultType[jj] += match.get(kk)/(len(data)*(dimension-1))

    maxPotential = 0        #保存最大可能性
    maxPotentialIndex = -1  #保存最大可能性的结果
  
    for kk in resultType.keys():
        #计算每种结果的可能性
        temp = 1
        for i in range(dimension-1):
            temp *= match[(t[i],kk)]/len(data)
        temp /= resultType[kk]
        if temp > maxPotential:
            maxPotential = temp
            maxPotentialIndex = kk 

    #print(resultType)
    #print(match)
    #print(maxPotential,maxPotentialIndex)
    return maxPotentialIndex,maxPotential


def test(t):
    """
    测试函数,自变量为z的情况下的预测结果

    Args:
        t:测试的样本
    
    Returns:
        result:测试结果
    """
    return

data = loadData('data.txt')
print(bayes(data,[2,'S']))
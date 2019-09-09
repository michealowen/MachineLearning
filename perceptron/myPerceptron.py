"""
Author: michealowen
Last edited: 2019.7.15,Monday
实现感知机算法
"""
#coding=UTF-8

import numpy as np
import matplotlib.pyplot as plt
import random

def loadData(fileName):
    """
    载入数据.
    
    Args:
        fileName:文件名
    
    Returns:
        data:数据,ndarray类型
    """
    data = np.loadtxt(fileName,delimiter=',')
    return data

def update(data):
    """
    不断迭代,更新直线的向量
    W1x1+W2x2+.....+Wnxn+W(n+1)Y+b=0

    Args:
        data:参与运算的数据
    
    Returns:
        w:直线的方向向量
        b:常数项的系数
    """
    dimension = len(data[0])-1
    w = [ 0 , 1 ]  
    b = 0
    n = 0.1
    error_data = []
    #首先选出误分类点
    for i in range(len(data)):
        if data[i][dimension]*judge(np.append(data[i][0:2],[1]),w+[b]) < 0:
            error_data.append(data[i])

    while len(error_data) > 0:
        #随机选取一个点进行SGD
        print(len(error_data))
        index = random.randint(0,len(error_data)-1)
        w += np.dot(n*error_data[index][dimension],error_data[index][0:dimension])
        b += n*error_data[index][dimension]

        error_data.clear()

        for i in range(len(data)):
            if data[i][dimension]*judge(np.append(data[i][0:2],[1]),w.tolist()+[b]) < 0:
                error_data.append(data[i])
    print(w,b)
    return (w,b)

def judge(t,e):
    """
    判断点t位于直线的上方还是下方

    Args:
        t:点的向量
        e:直线的向量

    Returns:
        1:表示位于上方
        -1:表示位于下方
        0:表示位于直线上
    """
    if np.dot(t,e) > 0:
        return 1
    elif np.dot(t,e) < 0:
        return -1
    else:
        return 0

def show(data,w,b):
    """
    绘制点和线
    """
    for i in range(len(data)):
        if data[i][2] > 0:
            plt.plot(data[i][0],data[i][1],'ok')
        else:
            plt.plot(data[i][0],data[i][1],'or')
    x = np.arange(-100,100,5)
    y = -w[0]/w[1]*x - b/w[1]
    plt.plot(x,y)
    plt.show()
    return None

data = loadData('data.txt')
w,b = update(data)
show(data,w,b)
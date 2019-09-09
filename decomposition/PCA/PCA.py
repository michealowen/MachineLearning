"""
Author: Michealowen
Last edited:2019.7.26,Friday
PCA降维算法
"""
#encoding=UTF-8

from sklearn import datasets
from sklearn.utils import Bunch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def loadData():
    '''
    Returns:
        data:数据矩阵
    '''
    data  = datasets.load_iris().data
    lable = datasets.load_iris().target    # 数据类类别        

    return data,lable

def PCA(data,dim):
    '''
    对数据进行降维 n -> dim
    '''
    #首先确定 样本个数m和维数n
    m = data.shape[0]
    cov = data
    #将数据中心化
    cov -= np.mean(cov,0)
    #再计算协方差矩阵
    cov = 1/(m-1)*np.dot(cov.T,cov)
    print("协方差矩阵:")
    print(cov)
    #求协方差矩阵的特征值,特征向量
    eigenValue,eigenVector = np.linalg.eig(cov)
    print("特征值:")
    print(eigenValue)
    print("特征向量:")
    print(eigenVector)
    #将特征值进行topK排序
    eigenValueK = np.argpartition(eigenValue,-dim)[-dim:]  #保存的是特征值的下标
    print("最大的K个特征值的下标:")
    print(eigenValueK)
    #扩充一个维度
    eigenValueK = eigenValueK[np.newaxis,:]
    eigenValueKValue = np.array([eigenValue[i] for i in eigenValueK])  #填充的一行为下标对应的特征值
    eigenValueK = eigenValueK.astype('float')
    eigenValueK = np.insert(eigenValueK,1,eigenValueKValue,axis=0)
    eigenValueK = eigenValueK[:,np.argsort(-eigenValueK[1])]   #按照第二列的特征值真值排序
    #第一行为特征向量的读取顺序
    print(eigenValueK)  

    #将特征向量取出并转置
    P = np.array([eigenVector[i] for i in eigenValueK[0].astype(int)])
    print("K特征向量为:")
    print(P)
    data = np.dot(data,P.T)
    print("降维后矩阵的形状")
    print(data.shape)
    return data

def show(data,lable):
    '''
    绘制降维后的数据
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    lableType = np.unique(lable)
    for i in range(len(data)):
        if lable[i] == lableType[0]:
            ax.scatter(data[i][0], data[i][1], data[i][2], c='k', marker='o')
            #ax.plot(data[i][0],data[i][1],data[i][2],'ok')
        elif lable[i] == lableType[1]:
            ax.scatter(data[i][0], data[i][1], data[i][2], c='r', marker='o')
            #ax.plot(data[i][0],data[i][1],data[i][2],'or')
        else:
            ax.scatter(data[i][0], data[i][1], data[i][2], c='b', marker='o')
            #ax.plot(data[i][0],data[i][1],data[i][2],'ob')
    plt.show()
    return None

'''
data,lable = loadData()
show(PCA(data,3),lable)
'''

"""
Author: michealowen
Last edited: 2019.1.3,Monday
Fisher线性判别,LDA(Linear Discrimination Analysis)
使用乳腺癌数据集测试(二分类)
"""
#encoding=UTF-8

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class Fisher:
    '''
    Fisher模型
    '''

    def __init__(self,X,x_test,Y,y_test):
        '''
        Params:
            X:样本,shape=(m,n)
            Y:为标注,shape=(1,m)
            x_test:测试集的数据
            y_test:测试集的标签
        '''
        self.X=X
        #将样本变为增广样本
        #self.X = np.append(self.X,np.array([[1 for i in range(len(X))]]).T,1)
        #print(self.X.shape)
        self.Y=Y
        n = self.X.shape[1]
        #self.w = np.array([0 for i in range(n+1)],dtype='float64')
        self.x_test = x_test
        self.y_test = y_test

    def __split(self):
        '''
        将数据按照标签分成两类
        '''
        self.x_1 = self.X[np.argwhere(self.Y==0)]    #健康
        self.x_1 = self.x_1.reshape(self.x_1.shape[0],self.x_1.shape[2])
        self.x_2 = self.X[np.argwhere(self.Y==1)]    #不健康
        self.x_2 = self.x_2.reshape(self.x_2.shape[0],self.x_2.shape[2])
        return None
 
    def __getMeanVector(self):
        '''
        计算均值向量
        '''
        self.mean_1 = np.mean(self.x_1,axis=0) 
        self.mean_2 = np.mean(self.x_2,axis=0)
        return None

    def __getWithinClassScatter(self):
        '''
        计算类内离散度矩阵,shape=(n,n)
        '''
        n = self.X.shape[1]
        s1 = np.array([[0 for i in range(n)] for i in range(n)],dtype='float64')
        s2 = np.array([[0 for i in range(n)] for i in range(n)],dtype='float64')   
        for i in range(len(self.x_1)):
            #print(np.dot(self.x_1[i].T,self.x_1[i]))
            s1 += np.dot(self.x_1[i].reshape(n,1),self.x_1[i].reshape(1,n))

        #print(s1)

        for i in range(len(self.x_2)):
            s2 += np.dot(self.x_2[i].reshape(n,1),self.x_2[i].reshape(1,n))

        #print(s2)

        self.withinClassScatterMatrix = s1 + s2
        return None   

    def __getWeight(self):
        '''
        计算投影方向
        '''
        print(self.withinClassScatterMatrix)
        '''
        u, s, v = np.linalg.svd(self.withinClassScatterMatrix)  # 奇异值分解
        s_w_inv = np.dot(np.dot(v.T, np.linalg.inv(np.diag(s))), u.T)
        self.w = np.dot(s_w_inv,self.mean_1-self.mean_2)
        '''
        self.w = np.linalg.solve(self.withinClassScatterMatrix,self.mean_1-self.mean_2)
        return None

    def train(self):
        '''
        训练过程
        '''
        self.__split()
        self.__getMeanVector()
        self.__getWithinClassScatter()
        self.__getWeight()

    def getPrecision(self):
        '''
        计算准确率
        '''
        #绘制分界情况
        plt.scatter(np.dot(self.x_1,self.w.T),np.array([0 for i in range(len(self.x_1))]),c='red',marker='v')
        plt.scatter(np.dot(self.x_2,self.w.T),np.array([0 for i in range(len(self.x_2))]),c='blue',marker='o')
        plt.legend(['benign','malignant'])
        #计算准确率
        center_1 = np.dot(self.mean_1,self.w.T)
        center_2 = np.dot(self.mean_2,self.w.T)      
        correct_num = 0
        for i in range(len(self.x_test)):
            temp = np.dot(self.x_test[i],self.w.T)
            if np.abs(temp-center_1) < np.abs(temp-center_2) and self.y_test[i] == 0:
                correct_num+=1
            if np.abs(temp-center_1) > np.abs(temp-center_2) and self.y_test[i] == 1:
                correct_num+=1
        print(correct_num/len(self.y_test))
        plt.show()
        return None


if __name__ == '__main__':
    data = load_breast_cancer()
    x = data.data
    y = data.target
    X_train, X_test, y_train, y_test=train_test_split(x,y,test_size=0.8)
    model = Fisher(X_train, X_test, y_train, y_test)
    model.train()
    model.getPrecision()
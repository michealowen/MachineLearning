"""
Author: michealowen
Last edited: 2019.11.1,Friday
线性回归算法,使用波士顿房价数据集
"""
#encoding=UTF-8

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

class linearRegression:
    '''
    线性回归模型类
    '''
    
    def __init__(self,X,x_test,Y,y_test):
        '''
        X为样本,shape=(m,n),Y为标注,shape=(1,m)
        '''
        self.X=X
        self.Y=Y
        self.x_test=x_test
        self.y_test=y_test
        n = self.X.shape[1]
        self.w = np.array([0 for i in range(n+1)],dtype='float64')

    def preProcess(self,x):
        '''
        加上x0=1,并对样本进行归一化
        ''' 
        x = np.c_[x,np.array([1 for i in range(x.shape[0])])]
        for i in range(x.shape[1]-1):
            x[:,i] = (x[:,i] - np.mean(x[:,i]))/np.std(x[:,i])
        return x

    def fit(self,method='GBD',alpha=None,iterNums=None,batchSize=None):
        '''
        使用传入的method参数,选择适当的拟合方法
        GBD:梯度下降,SGD:随机梯度下降,SBGD:小批量梯度下降,MT:矩阵法求解
        '''
        self.X = self.preProcess(self.X)   #预处理数据
        if method == 'BGD':
            self.BGD(alpha,iterNums)
        elif method == 'SGD':
            self.SGD(alpha)
        elif method == 'MT':
            self.MT()
        elif method == 'SBGD':
            self.SBGD(alpha)
        pass

    def MT(self):
        """
        基于矩阵求解的方法
        """
        self.w = np.dot(np.linalg.pinv(self.X),self.Y.T).T
        m,n = self.X.shape   #m为样本数,n为维度
        J = 1/m * np.dot( (np.dot(self.X,self.w.T) - self.Y),(np.dot(self.X,self.w.T) - self.Y).T)  #MSE
        print(J)
        return None

    def BGD(self,alpha,iterNums):
        '''
        使用所有样本进行梯度下降
        '''
        if alpha == None:
            print('缺少参数:迭代步长')
        if iterNums == None:
            print('缺少参数:迭代次数')  
        m,n = self.X.shape   #m为样本数,n为维度
        #w = np.array([0 for i in range(n)],dtype='float64')
        i = 0
        MinCost = float("inf")
        #while i<iterNums:
        while True:
            J = 1/m * np.dot( (np.dot(self.X,self.w.T) - self.Y),(np.dot(self.X,self.w.T) - self.Y).T)
            '''
            if J < MinCost:
                MinCost = J
                print(J," ",i)
                self.w -= 2/m * alpha * (np.dot(self.X.T ,(np.dot( self.X ,self.w.T) - self.Y.T ))).T 
                i += 1 
            '''
            if J > np.mean(self.Y)*0.1:
                print(J," ",i)
                self.w -= 2/m * alpha * (np.dot(self.X.T ,(np.dot( self.X ,self.w.T) - self.Y.T ))).T 
                i += 1 
            else:
                break             
        return None

    def SGD(self,alpha):
        '''
        随机梯度下降
        '''
        if alpha == None:
            print('缺少参数:迭代步长')
        m,n = self.X.shape   #m为样本数,n为维度
        #w = np.array([0 for i in range(n)],dtype='float64')
        i = 0
        MinCost = float("inf")
        while True:
            partIndex = np.random.randint(len(self.X))
            X_part = self.X[partIndex]
            Y_part = self.Y[partIndex]
            J = 1/m * np.dot( (np.dot(X_part,self.w) - Y_part),(np.dot(X_part,self.w) - Y_part).T)
            if abs(J - MinCost) < 0.0001:
                break
            else:
                print("J:",J," ",i)
                self.w -= 2/m * alpha * (np.dot(X_part.T ,(np.dot( X_part ,self.w.T) - Y_part.T ))).T 
                i = i+1
                MinCost = J
        return None

    def SBGD(self,alpha):
        '''
        小批量梯度下降
        '''
        if alpha == None:
            print('缺少参数:迭代步长')
        m,n = self.X.shape   #m为样本数,n为维度

        i = 0
        MinCost = float("inf")
        while True:
            partIndex = np.random.choice(range(m),int(m/10))
            X_part = self.X[partIndex]
            Y_part = self.Y[partIndex]
            J = 1/m * np.dot( (np.dot(X_part,self.w) - Y_part),(np.dot(X_part,self.w) - Y_part).T)
            if abs(J - MinCost) < 0.0001:
                break
            else:
                print("J:",J," ",i)
                self.w -= 2/m * alpha * (np.dot(X_part.T ,(np.dot( X_part ,self.w.T) - Y_part.T ))).T 
                i = i+1
                MinCost = J
        return None

    def predict(self,data):
        '''
        预测输入数据对应的输出
        '''
        data = self.preProcess(data)
        y = np.dot(data,self.w)
        print(y)
        return None

def evaluate(w,x_test,y_test):
    '''
    通过测试集评估模型的好坏,计算RSS(sum of squares for errors)
    '''
    print('评估')
    print(np.sum(np.square((np.dot(x_test,w.T)-y_test)))/len(x_test))
    return None


if __name__ == '__main__':

    boston = load_boston()
    x_train,x_test,y_train,y_test= train_test_split(boston.data,boston.target,test_size=0.1,random_state=0)
    model = linearRegression(x_train,x_test,y_train,y_test)
    #model.fit('BGD',alpha=0.1,iterNums=10000)
    model.fit('SBGD',alpha=0.1,iterNums=10000)
    evaluate(model.w,model.preProcess(x_test),y_test)

    #sklearn的线性回归
    from sklearn.linear_model import LinearRegression
    skmodel = LinearRegression()
    skmodel.fit(x_train,y_train)
    print(skmodel.score(x_test,y_test))
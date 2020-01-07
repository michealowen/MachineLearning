"""
Author: michealowen
Last edited: 2019.12.30,Monday
K均值聚类
"""
#encoding=UTF-8

import numpy as np
import matplotlib as mb
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from collections import Counter

class Kmeans:
    def __init__(self,X,k):
        self.X = X
        self.k = k
        pass

    def Euclid_dis(self, x, y):
        '''
        通过矩阵计算样本矩阵的欧氏距离
        '''
        x_length  = x.shape[0]
        y_length  = y.shape[0]
        dists = np.zeros((x_length, y_length))
        # because(X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train, so
        d1 = -2 * np.dot(x, y.T)    # shape (num_test, num_train)
        d2 = np.sum(np.square(x), axis=1, keepdims=True)    # shape (num_test, 1)
        d3 = np.sum(np.square(y), axis=1)     # shape (num_train, )
        #print(d1.shape,d2.shape,d3.shape,(d1+d2+d3).shape)
        dists = d1 + d2 + d3  # broadcasting
        #return np.sqrt(d1 + d2 + d3)
        return np.sqrt(dists)

    def train(self):
        '''
        训练
        '''
        self.select_point()
        self.adjust_point()
        pass

    def select_point(self):
        '''
        #确定初始的中心点，从原始数据中选出彼此距离最远的k个点
        '''
        data = self.X
        m = len(self.X)

        #首先随机选出一个点
        index  = np.random.randint(0,m-1)
        self.centers_index = np.array([index])
        #计算欧式距离矩阵mtx
        mtx = self.Euclid_dis(data,data)

        #不断将点选入s
        temp = np.zeros((1,m))
        #while len(s) < k:
        while len(self.centers_index) < self.k:
            max_dis = 0
            max_dis_index=-1
            for i in self.centers_index:
                temp += mtx[i,:]
            for i in range(m):          

                if temp[0][i] > max_dis and i not in self.centers_index:
                    max_dis = temp[0][i]
                    max_dis_index = i
            self.centers_index = np.append(self.centers_index,max_dis_index)
            temp = np.zeros((1,m))
        return None

    def adjust_point(self):
        '''
        #不断迭代，调整中心点位置
        '''
        data = self.X
        m = len(data)

        #1.确定初始的中心的位置
        self.centers = data[self.centers_index].copy()
        
        #2.计算距离矩阵
        for i in range(10000):

            #迭代一百次
            mx = self.Euclid_dis(self.centers,data) 
            #3.进行分类
            #List_class为每一类中包含的点
            List_class = [[] for i in range(self.k)]
            for j in range(m):
                index = np.argmin(mx[:,j])
                List_class[index].append(j)
                
            #4.重新确定中心
            for i in range(self.k):
                '''
                要确定k个中心
                '''
                #coordinate_sum为所有点的坐标之和
                coordinate_sum = 0
                for j in range(len(List_class[i])):
                    coordinate_sum += data[List_class[i][j]] 
                if len(List_class[i]) > 0:
                    self.centers[i] = coordinate_sum/len(List_class[i])
            self.List_class = List_class
        #print(List_class)
        return None

def show(centers,List_class,data,k):
    mark = ['b','g','r','y','k','c','m']
    for i in range(k):
        plt.plot(centers[i][0],centers[i][1],'x'+mark[i])
        for j in range(len(List_class[i])):
            plt.plot(data[List_class[i][j]][0],data[List_class[i][j]][1],'o'+mark[i])
        
    plt.show()
    return None

def getPrecision(data,labels,centers,List_class):
    '''
    计算准确率
    '''
    #先对类别进行映射  centers共k个,对应lables的k个

    centers_labels = []
    for i in range(len(List_class)):
        centers_labels.append(Counter(labels[List_class[i]]).most_common()[0][0])
    print(centers_labels)

    error_num = 0
    for i in range(len(List_class)):
        for j in range(len(List_class[i])):
            #print(List_class[i][j])
            if labels[int(List_class[i][j])] != centers_labels[i]:
                error_num += 1 
    print('准确率为',1-error_num/len(labels))
    pass

def loadData(fileName):
    '''
    载入数据
    '''
    data = np.loadtxt( fileName , delimiter=',')
    return data
    
if __name__ == '__main__':
    '''
    data = loadData('data.txt')
    '''
    #使用iris数据集
    data = load_iris()
    labels = data.target
    data = data.data

    model = Kmeans(data,4)
    model.train()
    #show(model.centers,model.List_class,data,4)
    getPrecision(data,labels,model.centers,model.List_class)
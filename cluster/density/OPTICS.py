"""
Author: michealowen
Last edited: 2019.10.19,Saturday
OPTICS聚类,使用sklearn生成数据集
"""
#encoding=UTF-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.neighbors import KDTree
from sklearn.cluster import OPTICS
from queue import Queue
from time import time


class optics:

    def __init__(self,eps,minPts):
        '''
        构造函数,参数分别为邻域半径和最小核心点邻居数量
        '''
        self.eps = eps
        self.minPts = minPts
        return None

    def getCoreIndex(self,X):
        '''
        获取核心点的下标,数据集为X
        '''
        #1. 构造KD树
        self.kd = KDTree(X)   #10为节点内的样本数,若为len(x),则KD树退化成蛮力求解,即遍历
        self.coreIndex = np.array([])
        for i in range(len(X)):
            #print(len(self.kd.query_radius(X[i:i+1],r=self.eps)))
            if len(self.kd.query_radius(X[i:i+1],r=self.eps)[0]) >= self.minPts:
                self.coreIndex = np.append(self.coreIndex,i)
        return None

    def getCoreDist(self,x):
        '''
        得到某点成为核心点的最短半径
        '''
        t = self.kd.query([x],k=self.minPts)[0]  #方法返回tuple,tuple[0]为距离,tuple[1]为下标
        return t[0,len(t[0])-1]

    def getDist(self,x1,x2):
        '''
        返回欧氏距离
        '''
        return np.sqrt(np.sum(np.square(x1-x2)))


    def getReachDist(self,x,corePoint):
        '''
        获得可达距离,max(成为核心的最小半径,核心点与点的距离)
        '''
        #print(max(self.getCoreDist(corePoint),self.getDist(x,corePoint)))
        return max(self.getCoreDist(corePoint),self.getDist(x,corePoint))

    def resort(self):
        '''
        根据可达距离,对有序队列进行排序
        '''
        orderedReachDist = self.reachDistList[self.orderedList]
        self.orderedList = self.orderedList[np.argsort(orderedReachDist)]
        return None

    def fit(self,X):
        '''
        对数据集X进行聚类,算法维护一个未处理点的集合,一个有序队列,一个输出队列(都存储的点在数据集中的下标)
        未处理点集合和输出点集合互斥,进入输出点集合后要从未处理点中删除
        还需要一个可达距离列表
        '''
        #1.首先将点的下标全部加入未处理点集合D
        self.D = np.arange(0,len(X),1)
        self.outputList = np.array([],dtype=int)
        self.orderedList = np.array([],dtype=int)
        self.reachDistList = np.array([1 for i in range(len(X))],dtype=float)
        self.getCoreIndex(X)

        j=0
        #2.不断从D中取出一点
        while len(self.D) > 0:
            print(len(self.D))
            j+=1
            p = np.random.randint(len(self.D))
            p = self.D[p]
            self.outputList = np.append(self.outputList,p)
            self.D = np.delete(self.D,np.argwhere(self.D==p))
            print('从D中删除',p)
            #将该点加入输出队列,其邻域内所有点加入有序队列
            if p in self.coreIndex:
                #print(self.kd.query_radius(X[p:p+1],self.eps)[0])
                for i in self.kd.query_radius(X[p:p+1],self.eps)[0]:
                    if i in self.D and i!=p:
                        if i not in self.orderedList:
                            #print('有序队列加入',i)
                            self.orderedList = np.append(self.orderedList,i)
                            self.reachDistList[i] = self.getReachDist(X[i],X[p])
                            #print('可达距离为',self.reachDistList[i])
                        else:
                            if self.reachDistList[i] > self.getReachDist(X[i],X[p]):
                                #若可达距离变小,则进行更新
                                #print('更新可达距离')
                                self.reachDistList[i] = self.getReachDist(X[i],X[p])
                                #print('可达距离更新为',self.reachDistList[i])
            self.resort()
            #处理有序队列
            while len(self.orderedList) > 0:
                print('有序队列长度',len(self.orderedList))
                print(self.reachDistList[self.orderedList])
                q = self.orderedList[0]
                self.outputList = np.append(self.outputList,q)
                self.orderedList = np.delete(self.orderedList,np.argwhere(self.orderedList==q))
                self.D = np.delete(self.D,np.argwhere(self.D==q))
                #print('取出一点',q)
                if q in self.coreIndex:
                    for i in self.kd.query_radius(X[q:q+1],self.eps)[0]:
                        if i in self.D and i!=q:
                            if i not in self.orderedList:
                                #print(i,'加入有序队列')
                                self.orderedList = np.append(self.orderedList,i)
                                self.reachDistList[i] = self.getReachDist(X[i],X[q])

                            else:
                                if self.reachDistList[i] > self.getReachDist(X[i],X[q]):
                                    #若可达距离变下,则进行更新
                                    #print('更新可达距离')
                                    self.reachDistList[i] = self.getReachDist(X[i],X[q])
                self.resort()
        print(j)
        for i in range(len(X)):
            if self.reachDistList[i] == -0.1:
                j-=1
        print(j)       
        return None

    def show(self):
        '''
        绘制OPTICS聚类结果,横轴为输出队列,纵轴为可达距离
        '''
        fig = plt.figure(figsize=(10,10))
        plt.plot(np.arange(len(self.outputList)),self.reachDistList[self.outputList])
        plt.show()
        return None


if __name__ == '__main__':
    
    
    x1,y1 = datasets.make_moons(n_samples=100,shuffle=True,noise=0.05,random_state=None)
    x2,y2 = datasets.make_blobs(n_samples=100,n_features=2,centers=[[2,2],[5,6]],cluster_std=[[0.1],[0.1]],random_state=0)
    C1 = [-5, -2] + .8 * np.random.randn(100, 2)
    C2 = [4, -1] + .1 * np.random.randn(100, 2)
    C3 = [1, -2] + .2 * np.random.randn(100, 2)
    C4 = [-2, 3] + .3 * np.random.randn(100, 2)
    C5 = [3, -2] + 1.6 * np.random.randn(100, 2)
    
    #x = np.concatenate((C1,C2,C3,C4,C5))
    x = np.concatenate((x1,x2))
    np.random.shuffle(x)
    
    #使用自己实现的OPTICS
    plt.scatter(x[:,0],x[:,1])
    plt.show()
    op = optics(5,2)
    op.fit(x)
    op.show()
    
    '''
    sklearn的OPTICS算法
    plt.scatter(x[:,0],x[:,1])
    plt.show()
    oop = OPTICS(min_samples=5,max_eps=5)
    oop.fit(x)
    print(oop.ordering_)
    plt.plot(np.arange(len(oop.ordering_)),oop.reachability_[oop.ordering_])
    plt.show()
    '''
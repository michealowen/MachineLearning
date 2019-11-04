"""
Author: michealowen
Last edited: 2019.9.20,Friday
DBSCAN聚类,使用sklearn生成数据集
"""
#encoding=UTF-8

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from queue import Queue
from time import time
from scipy.spatial import KDTree

def genData():
    '''
    使用sklear生成数据集
    '''
    x1,y1 = datasets.make_moons(n_samples=1000,shuffle=True,noise=0.05,random_state=None)
    x2,y2 = datasets.make_blobs(n_samples=1000,n_features=2,centers=[[2,2]],cluster_std=[[0.1]],random_state=5)
    x = np.concatenate((x1,x2))
    y = np.concatenate((y1,y2))
    return x,y

def show(cluster,data):
    '''
    使用plt绘制数据    
    '''
    color = np.array(['r','c','m','b','y','k','g'])
    for i in range(len(cluster)):
        plt.scatter(data[cluster[i]][:,0],data[cluster[i]][:,1],s=8,c=color[i])
    
    for i in range(len(data)):
        flag = False
        for j in range(len(cluster)):
            if i in cluster[j]:
                flag = True
                break
        if flag == False:
            plt.plot(data[i,0],data[i,1],'ok')
    plt.show()

def Eucl_dis(x,y):
    '''
    计算两个点的欧几里得距离
    '''
    return np.sqrt(np.sum((x-y)**2))

def getDistMatrix(data):
    '''
    计算距离矩阵
    Args:
        data:点的坐标
    Returns:
        M:距离矩阵
    '''
    length = len(data)
    M = np.array([[ 0 for i in range(length)] for i in range(length)])
    M = M/1.0
    for i in range(length):
        for j in range(i+1,length):
            M[i,j] = Eucl_dis(data[i],data[j])
            M[j,i] = M[i,j]
    return M

def myDBSCAN(data,Eps,minPts):
    '''
    DBSCAN算法
    Args:
        Eps:标准半径
        minPts:最少包含点数
    '''
    M = getDistMatrix(data)
    m = len(M)
    core = np.array([],dtype=int)
    border = {}
    noise = np.array([],dtype=int)
    # 1.标记点
    for i in range(m):
        num=0
        for j in range(m):
            if i == j:
                continue
            elif M[i,j] <= Eps:
                num+=1
        if num >= minPts:
            #此点为核心点
            core = np.append(core,i)
            for j in range(m):
                if i != j:
                    if M[i,j] <= Eps and j not in core:
                        # 加入边界点
                        if i not in border.keys():
                            border[i] = np.array([j])
                        else:
                            if j in noise:
                                noise = np.delete(noise,np.argwhere(noise==j))
                            border[i] = np.append(border[i],j)

        else:
            #若此点此时还不是边界点,则先加入噪音点.
            #若该点不是噪音点,则在之后可将该点从噪音点还原为边界点
            noise = np.append(noise,i)
    #2.密度可达core点合并,BFS
    #cluster为簇
    cluster={}
    clusterNum=0
    #visited保存core点的访问信息
    visited = np.array([False for i in range(len(core))])
    #pointQ为队列
    pointQ = Queue()

    while np.all(visited==True) == False:
        for i in range(len(core)):
            if visited[i] == False:
                visited[i] == True
                pointQ.put(core[i])
                cluster[clusterNum]=np.array([core[i]],dtype=int)
                break
        while pointQ.empty() == False:
            currentCore=pointQ.get()
            #visited[np.argwhere(core==currentCore)]=True
            for i in range(len(core)):
                if core[i] != currentCore and visited[i] == False:
                    if M[currentCore,core[i]] <= Eps:
                        #加入队列
                        pointQ.put(core[i])
                        visited[i]=True
                        #加入簇
                        cluster[clusterNum] = np.append(cluster[clusterNum],core[i])
        clusterNum+=1
    
    #3.把border点合并到簇
    for it in border.items():
        for i in range(clusterNum):
            if it[0] in cluster[i]:
                cluster[i] = np.append(cluster[i],it[1])

    return cluster

def myKDDBSCAN(data,Eps,minPts):
    '''
    DBSCAN算法(使用KD树改进)
    Args:
        Eps:标准半径
        minPts:最少包含点数
    '''
    kd = KDTree(data)
    #M = getDistMatrix(data)
    #m = len(M)
    core = np.array([],dtype=int)
    border = {}
    noise = np.array([],dtype=int)
    # 1.标记点
    for i in range(len(data)):
        N = kd.query_ball_point(data[i],r=Eps) 
        if len(N) >= minPts:
            #该点为core
            core = np.append(core,i)
            #将该点的紧邻加入border
            for j in N:
                if i not in border.keys():
                    border[i] = np.array([j])
                else:
                    if j in noise:
                        noise = np.delete(noise,np.argwhere(noise==j))
                        border[i] = np.append(border[i],j)

        else:
            #若此点此时还不是边界点,则先加入噪音点.
            #若该点不是噪音点,则在之后可将该点从噪音点还原为边界点
            noise = np.append(noise,i)
    #2.密度可达core点合并,BFS
    #cluster为簇
    cluster={}
    clusterNum=0
    #visited保存core点的访问信息
    visited = np.array([False for i in range(len(core))])
    #pointQ为队列
    pointQ = Queue()

    while np.all(visited==True) == False:
        for i in range(len(core)):
            if visited[i] == False:
                visited[i] == True
                pointQ.put(core[i])
                cluster[clusterNum]=np.array([core[i]],dtype=int)
                break
        while pointQ.empty() == False:
            currentCore=pointQ.get()
            visited[np.argwhere(core==currentCore)]=True
            N = kd.query_ball_point(data[int(currentCore)],r=Eps)  
            for i in N:
                #若未访问过,加入队列
                index = np.argwhere(core==i)
                #print(index)
                if visited[index] == False:
                    pointQ.put(i)
                    visited[index]=True
                    cluster[clusterNum] = np.append(cluster[clusterNum],core[index])
        clusterNum+=1
    
    print(clusterNum)
    #3.把border点合并到簇
    for it in border.items():
        for i in range(clusterNum):
            if it[0] in cluster[i]:
                cluster[i] = np.append(cluster[i],it[1])

    return cluster

if __name__ == '__main__':
    x,y=genData()
    t = time()
    cluster=myDBSCAN(x,0.1,10)
    print(time()-t)
    #1000个数据,原始DBSCAN24秒
    show(cluster,x)
    t = time()
    cluster=myKDDBSCAN(x,0.1,10)
    print(time()-t)
    #1000个数据,KD树改进DBSCAN12秒
    show(cluster,x)
    t = time()
    y=DBSCAN(0.1,10,'euclidean').fit_predict(x)
    print(time()-t)
    #1000个数据sklearn的DBSCAN0.03秒
    plt.scatter(x[:, 0], x[:, 1], c=y)
    plt.show()
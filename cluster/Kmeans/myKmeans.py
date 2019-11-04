import numpy as np
import matplotlib as mb
import matplotlib.pyplot as plt
import random

#载入数据
def loadData(fileName):
    data = np.loadtxt( fileName , delimiter=',')
    return data


#计算欧几里得距离
def Eucl_dis(x,y):
    return np.sqrt(np.sum((x-y)**2))


#确定初始的中心点，从原始数据中选出彼此距离最远的k个点
def select_point(k,m):
        data = loadData('data.txt')
        #首先随机选出一个点
        index  = random.randint(0,m-1)
        
        s = np.array([index])
        
        #计算欧式距离矩阵mtx
        mtx = np.zeros((m,m))
        for i in range(m):
                for j in range(m):
                        if i != j:
                                mtx[i][j] = Eucl_dis(data[i,:],data[j,:])

        #不断将点选入s
        temp = np.zeros((1,m))
        while len(s) < k:
                max_dis = 0
                max_dis_index=-1
                for i in s:
                        temp += mtx[i,:]
                for i in range(m):          
                #s = np.append(s,np.argmax(temp))
                        if temp[0][i] > max_dis and i not in s:
                                max_dis = temp[0][i]
                                max_dis_index = i
                s = np.append(s,max_dis_index)
                temp = np.zeros((1,m))
    
        return (s,mtx)

#不断迭代，调整中心点位置

def adjust_point(s,m):
        data = loadData('data.txt')
        #1.确定初始的中心的位置
        k = 7
        #cent = {}
        cent = [[0,0] for i in range(k)]
        for i in range(len(s)):
                cent[i] = data[s[i]]

        #2.计算距离矩阵
        mx = np.zeros((k,m))
        for i in range(100):
                for j in range(k):
                        for r in range(m):
                                mx[j][r] = Eucl_dis(cent[j],data[r])
                
                #3.进行分类
                List_class = [[] for i in range(k)]
                for j in range(m):
                        index = mx[:,j].tolist().index(min(mx[:,j]))
                        List_class[index].append(j)
                
                #4.重新确定中心
                for i in range(k):
                        sum_cen = 0
                        for j in range(len(List_class[i])):
                                sum_cen += data[List_class[i][j]] 
                        if len(List_class[i]) > 0:
                                cent[i] = sum_cen/len(List_class[i])

        return (cent,List_class)

def show(cent,List_class):
        k=7
        mark = ['b','g','r','y','k','c','m']
        for i in range(k):
                plt.plot(cent[i][0],cent[i][1],'x'+mark[i])
                for j in range(len(List_class[i])):
                        plt.plot(data[List_class[i][j]][0],data[List_class[i][j]][1],'o'+mark[i])
        
        plt.show()
        return None


data = loadData('data.txt')
[m,n] = data.shape
s = select_point(7,m)[0]
print(s)
[cent,List_class] = adjust_point(s,m)
show(cent,List_class)
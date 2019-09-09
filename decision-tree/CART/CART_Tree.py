"""
Author: michealowen
Last edited: 2019.8.7,Wednsday
CART算法生成决策树
"""
#encoding=UTF-8

import numpy as np
import pandas as pd
from collections import Counter

def loadData(fileName):
    data = pd.read_csv(fileName,delimiter=',')
    data = data.values
    return data

class Node:
    """
    CART树节点类
    """
    def __init__(self,isLeaf,leftChild,rightChild,nodeData,spilt,lable):
        """
        CART节点的构造函数

        Args:
            isLeaf:True为叶子节点,False为非叶节点
            leftChild:节点的左孩子
            rightChild:节点的右孩子
            nodeData:该节点的数据 
            spilt:在该节点分类的维度(叶节点此属性为-1)
            lable:叶子节点拥有类别,非叶子节点为None
        """
        self.isLeaf = isLeaf
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.nodeData = nodeData
        self.spilt = spilt
        self.lable = lable
        return

class CART_Tree:
    """
    CART树类,CART为二叉树
    """
    def __init__(self,data):
        """
        CART树的构造方法
        """
        self.root = self.buildTree(data,[i for i in range(len(data[0])-1)])
        return

    def buildTree(self,data,Artributes):
        """
        递归的生成树
        """
        dim = len(data[0])
        #若data中的标记相同,则返回叶子节点
        if len(np.unique(data[:,dim-1])) == 1:
            return Node(True,None,None,data,-1,data[0,dim-1])
        
        elif len(Artributes) == 0:
            return Node(True,None,None,data,-1,Counter(data[:,dim-1]).most_common(1)[0][0])
        
        #首先选出最优特征和最佳切分点
        minGini = float('inf') 
        minGiniIndexTupple = None
        for i in Artributes:
            for j in self.getConditionalGini(data,i).items():
                if j[1]< minGini:
                    minGini = j[1]
                    minGiniIndexTupple = j[0] 
               
        #然后将data按照区分点分为两部分
        leftData = []
        rightData = []
        for i in range(len(data)):
            if data[i,minGiniIndexTupple[0]] == minGiniIndexTupple[1]:
                leftData.append(data[i].tolist())
            else:
                rightData.append(data[i].tolist())

        #print(np.array(leftData))
        #print(np.array(rightData))
        return Node(False,self.buildTree(np.array(leftData),Artributes),self.buildTree(np.array(rightData),Artributes),data,minGiniIndexTupple[0],None)

    
    def getGini(self,data):
        """
        得到数据的基尼指数
        """
        if len(data) == 0:
            return 0
        dim = len(data[0])  #维度
        mapp = {}
        for i in range(len(data)):
            if mapp.get(data[i][dim-1]) == None:     #若data只剩下一行,就不能
                mapp[data[i][dim-1]] = 1
            else:
                mapp[data[i][dim-1]] += 1
               
        Gini = 1
        for i in mapp.values():
            Gini -= np.square(i/len(data))
        return Gini

    def getConditionalGini(self,data,index):
        """
        获得数据的条件基尼指数
        """
        if len(data) == 0:
            return 0
        #将数据在该维度分类
        mapp = {}
        for i in range(len(data)):
            if data[i][index] not in mapp.keys():
                mapp[data[i][index]] = np.array([data[i]])
            else:
                mapp[data[i][index]] = np.vstack((mapp[data[i][index]],data[i]))
        #print(mapp)
        #分类完成之后计算条件基尼指数
        conditionalGini = {}
        for kk in mapp.keys():
            #print(mapp.get(kk).tolist())
            temp = []
            for ii in data:
                if ii.tolist() not in mapp.get(kk).tolist():
                    #print(ii,"not in")
                    temp.append(ii.tolist())                   
            conditionalGini[(index,kk)] = len(mapp.get(kk))/len(data)*self.getGini(mapp.get(kk))+(1-len(mapp.get(kk))/len(data))*self.getGini(np.array(temp))       
        
        return conditionalGini

    def DLR(self,root):
        """
        递归先序遍历
        """
        #print(root.isLeaf,root.nodeData)
        if root.isLeaf == True:
            print("叶子节点","标签",root.lable,root.nodeData)
        else:
            print("非叶子节点","分割点",root.spilt,root.nodeData)
        if root.leftChild != None:
            self.DLR(root.leftChild)
        if root.rightChild != None:
            self.DLR(root.rightChild)
        return



data = loadData('loan_data.csv')
T = CART_Tree(data)
#print(T.getGini(data))
'''
print(T.getConditionalGini(data,0))
print(T.getConditionalGini(data,1))
print(T.getConditionalGini(data,2))
print(T.getConditionalGini(data,3))
'''
T.DLR(T.root)
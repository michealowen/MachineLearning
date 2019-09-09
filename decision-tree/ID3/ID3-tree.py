"""
Author: michealowen
Last edited: 2019.8.7,Wednsday
ID3算法生成决策树
"""
#coding=UTF-8

import pandas as pd
import numpy as np
from collections import Counter
from graphviz import Digraph

def loadData(fileName):
    """
    载入数据

    Args:
        fileName:数据文件名
    
    Returns:
        data:数组形式数据
    """
    data = pd.read_csv(fileName,delimiter=',',encoding='UTF-8')
    #data = data.values
    return data.values,data.columns.values

class ID3_Node:
    """
    ID3树的节点
    """

    def __init__(self,isLeaf,childList,nodeData,spilt,lable):
        """
        ID3节点的构造函数

        Args:
            isLeaf:True为叶子节点,False为非叶节点
            childList:该节点的子节点列表
            nodeData:该节点的数据 
            spilt:在该节点分类的维度(叶节点此属性为-1)
            lable:叶子节点拥有类别,非叶子节点为None
        """
        self.isLeaf = isLeaf
        self.childList = childList
        self.nodeData = nodeData
        self.spilt = spilt
        self.lable = lable
        return


class ID3_tree:
    """ID3树类"""

    def __init__(self,data,columns):
        """
        通过该数据集构造树
        Args:
            data数据
            columns:属性列表
        """
        self.data = data
        self.columns = columns
        if len(data) == 0:
            print("数据为空")
            return None 
        self.root = self.bulidTree(data,[i for i in range(len(data[0])-1)])
        return
    
    def Pre_trace(self,root):
        """
        先序遍历树
        """
        if root == None:
            return
        print(root.nodeData,'\n')
        if root.isLeaf == True:
            return
        else:
            for node in root.childList:
                self.Pre_trace(node)

    def buildMap(self,data):
        """
        通过data创建dict,Key为lable,Value为lable出现的次数
        """
        mapp = {}
        for i in range(len(data)):
            if mapp.get(data[i]) == None:
                mapp[data[i]] = 1
            else:
                mapp[data[i]] +=1
        return mapp

    def classifyData(self,data,index):
        """
        将data按照该维度分类
        Args:
            index:划分维度
        """
        mapp = {}
        for i in range(len(data)):
            if data[i,index] not in mapp.keys():
                mapp[data[i,index]] = [data[i]]
            else:
                mapp[data[i,index]] = np.vstack((mapp[data[i,index]],data[i]))
        return mapp

    def findMaxKey(self,mapp):
        """
        寻找mapp中,值最大的键
        """
        maxValue = 0
        maxKey = -1
        for i in mapp.items():
            if i[1] > maxValue:
                maxValue = i[1]
                maxKey = i[0]
        return maxKey

    def bulidTree(self,data,Artributes):
        """
        递归地构造树

        Args:
            data:进行构造的数据
        """
        dim = len(data[0])
        #如果所有样本都是同一类,则返回叶子节点
        #if len(set(data[:,dim-1])) == len(data):
        if len(np.unique(data[:,dim-1])) == 1: 
            return ID3_Node(True,None,data,-1,data[0,dim-1]) 
        #如果该点的特征值为空,则返回叶子节点,lable为data中最多的
        elif len(Artributes) == 0:
            #return ID3_Node(True,None,data,-1,self.findMaxKey(self.buildMap(data[:,dim-1])))
            return ID3_Node(True,None,data,-1,Counter(data[:,dim-1]).most_common(1)[0][0])

        #选取信息增益最大的维度,并根据该维度的取值将data分为若干子节点
        #选出维度
        entropy = self.getEntropy(data)
        print("熵为",entropy)
        maxGainValue = 0
        maxGainIndex = -1
        for i in Artributes:
            print(i)
            print("条件熵为",self.getConditionalEntropy(data,i))
            if entropy - self.getConditionalEntropy(data,i) > maxGainValue:
                maxGainIndex = i
                maxGainValue = entropy - self.getConditionalEntropy(data,i)
        #按照维度分类
        print("分类维度",maxGainIndex)
        Artributes.remove(maxGainIndex)
        nodeList = []

        for element in self.classifyData(data,maxGainIndex).values():
            #print(element)
            nodeList.append(self.bulidTree(element,Artributes))

        return ID3_Node(False,nodeList,data,maxGainIndex,None) 


    def getEntropy(self,data):
        """
        求出熵

        Args:
            data:数据集,最后一维参与运算

        Returns:
            entropy:熵
        """
        if len(data) == 0:
            print("数据为空")
            return
        
        dim = len(data[0])
        mapp = {}
        for i in range(len(data)):
            if mapp.get(data[i][dim-1]) == None:
                mapp[data[i][dim-1]] = 1
            else:
                mapp[data[i][dim-1]] +=1
        
        entrop = 0
        for i in mapp.keys():
            entrop -= mapp[i]/len(data)*np.log2(mapp[i]/len(data))

        return entrop


    def getConditionalEntropy(self,data,index):
        """
        求出条件熵
        
        Args:
            data:数据集
            index:条件的维度

        Returns:
            sumConditionEntrop:条件熵
        """
        if len(data) == 0:
            print("数据为空")
            return
        
        dim = len(data[0])
        sumConditionEntrop = 0
        #1.将data按照该维度分类,放入dict中
        #例如{'青年': [0, 1, 2, 3, 4], '中年': [5, 6, 7, 8, 9], '老年': [10, 11, 12, 13, 14]}
        indexCategory = {}
        for i in range(len(data)):
            if indexCategory.get(data[i][index]) == None:
                indexCategory[data[i][index]] = [i]
            else:
                indexCategory[data[i][index]].append(i)
        
        #2.将每一部分的数据分别求熵,按照比例相加得到条件熵
        for element in indexCategory.values():
            mapp = {}
            for j in element:
                if mapp.get(data[j][dim-1]) == None:
                    mapp[data[j][dim-1]] = 1
                else:
                    mapp[data[j][dim-1]] += 1
            
            for i in mapp.keys():
                sumConditionEntrop -= mapp[i]/len(element)*np.log2(mapp[i]/len(element))*len(element)/len(data)

        #print(indexCategory)
        #print(sumConditionEntrop)
        return sumConditionEntrop

    def buildGraphTree(self,currentNode,G):
        """
        使用Digraph可视化树,先序遍历
        """

        if currentNode.childList != None:
            for node in currentNode.childList:
                if node != None:
                    if node.isLeaf == True:
                        G.node(str(node.nodeData),str(node.nodeData),shape='box')
                    else:
                        G.node(str(node.nodeData),str(node.nodeData))
                    G.edge(str(currentNode.nodeData),str(node.nodeData),str(columns[currentNode.spilt])+str(np.unique(node.nodeData[:,currentNode.spilt])))
                    self.buildGraphTree(node,G)
        G.view()
        return None

root = ID3_Node(True,None,None,-1,None)
data,columns = loadData('loan_data.csv')
print(columns)
t = ID3_tree(data,columns)

#t.Pre_trace(t.root)
g = Digraph("loan")
g.node(str(t.root.nodeData),str(t.root.nodeData))
t.buildGraphTree(t.root,g)
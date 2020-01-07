"""
Author: michealowen
Last edited: 2020.1.02,Thursday
CART算法生成决策树
"""
#encoding=UTF-8

import numpy as np
import pandas as pd
from collections import Counter,defaultdict
from graphviz import Digraph

def loadData(fileName):
    data = pd.read_csv(fileName,delimiter=',')
    #data = data.values
    return data.values,data.columns.values

class Node:
    """
    CART树节点类
    """
    def __init__(self,isLeaf,leftChild,rightChild,nodeData,split,label):
        """
        CART节点的构造函数

        Args:
            isLeaf:True为叶子节点,False为非叶节点
            leftChild:节点的左孩子
            rightChild:节点的右孩子
            nodeData:该节点的数据 
            split:在该节点分类的维度(叶节点此属性为-1)
            label:叶子节点拥有类别,非叶子节点为None
        """
        self.isLeaf = isLeaf
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.nodeData = nodeData
        self.split = split
        self.label = label
        return

class CART_Tree:
    """
    CART树类,CART为二叉树
    """
    def __init__(self,data,columns):
        """
        CART树的构造方法
        """
        np.random.shuffle(data)
        self.data = data[:int(len(data)*1/10)]
        self.test_data = data[int(len(data)*1/10):]
        self.columns = columns
        self.root = self.buildTree(data,[i for i in range(len(data[0])-1)])
        self.REP_pruning()
        return

    def buildTree(self,data,Artributes):
        """
        递归的生成树
        """
        #dim = len(data[0])
        #若data中的标记相同,则返回叶子节点
        if len(np.unique(data[:,-1])) == 1:
            return Node(True,None,None,data,-1,data[0,-1])
        
        elif len(Artributes) == 0:
            return Node(True,None,None,data,-1,Counter(data[:,-1]).most_common(1)[0][0])
        
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
        mapp = defaultdict(int)
        for i in range(len(data)):
            mapp[data[i][-1]] += 1
               
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
            for d in data:
                if d.tolist() not in mapp.get(kk).tolist():
                    #print(ii,"not in")
                    temp.append(d.tolist())                   
            conditionalGini[(index,kk)] = len(mapp.get(kk))/len(data)*self.getGini(mapp.get(kk))+(1-len(mapp.get(kk))/len(data))*self.getGini(np.array(temp))       
        return conditionalGini

    def REP_pruning(self):
        '''
        后剪枝算法,使用准确率进行剪枝,Reduced-Error Pruning(REP,错误率降低剪枝）
        bottom-up,自底而上进行遍历
        '''
        #使用list来存储树结构,进行自底向上的遍历    首先将节点都加入栈中
        node_stack = [self.root]
        cursor = 0
        while len(node_stack) > cursor:
            if node_stack[cursor].isLeaf == False:
                if node_stack[cursor].leftChild != None:
                    node_stack.append(node_stack[cursor].leftChild)
                if node_stack[cursor].rightChild != None:
                    node_stack.append(node_stack[cursor].rightChild)                
            cursor+=1
        cursor-=1
        #不断出栈,进行剪枝,此时游标指向栈顶
        while cursor >= 0:
            #不断向上寻找非叶节点
            if node_stack[cursor].isLeaf == True:
                cursor-=1
            else:
                #找到非叶节点后,若在该节点剪枝后,准确率的到提升,则进行剪枝,使改节点成为叶节点
                p = self.evaluate()
                q = self.__pruned_evaluate(node_stack[cursor])
                print(p,q)
                if p < q:
                    print('准确率提升:',p,'->',q)
                    if node_stack[cursor].leftChild != None:
                        node_stack.remove(node_stack[cursor].leftChild)
                        node_stack[cursor].leftChild = None
                    if node_stack[cursor].rightChild != None:
                        node_stack.remove(node_stack[cursor].rightChild)
                        node_stack[cursor].rightChild = None
                    node_stack[cursor].isLeaf = True
                    node_stack[cursor].label = Counter(node_stack[cursor].nodeData[:,-1]).most_common(1)[0][0]
                    node_stack[cursor].split =-1
                cursor-=1
        return None

    def evaluate(self):
        '''
        测试模型的准确率
        Returns:
            模型在测试集上的准确率
        '''
        Sum = 0
        for i in self.test_data:
            #通过决策树得到预测结果
            if i[-1] == self.predict(i):
                Sum+=1
        return Sum/len(self.test_data)

    def predict(self,x):
        '''
        预测
        Params:
            x:预测数据
        '''
        flag = False
        temp_node = self.root
        while temp_node.isLeaf == False:
            flag = False
            if temp_node.leftChild != None:
                #左孩子
                if temp_node.leftChild.isLeaf == True:
                    if x[temp_node.split] == temp_node.leftChild.label:
                        flag = True
                        temp_node = temp_node.leftChild
                        continue
                elif x[temp_node.split] == temp_node.leftChild.nodeData[0,temp_node.split]:
                    #进到下一个分支
                    flag = True
                    temp_node = temp_node.leftChild
                    continue                      
            if temp_node.rightChild != None:
                #右孩子
                if temp_node.rightChild.isLeaf == True:
                    if x[temp_node.split] == temp_node.rightChild.label:
                        flag = True
                        temp_node = temp_node.rightChild
                        continue
                elif x[temp_node.split] == temp_node.rightChild.nodeData[0,temp_node.split]:
                    #进到下一个分支
                    flag = True
                    temp_node = temp_node.rightChild
                    continue   
            if flag == False:
                #如果没匹配到,少数服从多数
                return Counter(temp_node.nodeData[:,-1]).most_common(1)[0][0]
        return temp_node.label

    def __pruned_evaluate(self,node):
        '''
        测试剪枝树的准确率
        Returns:
            剪枝树在测试集上的准确率
        '''
        Sum = 0
        #print('测试数据',len(self.test_data))
        for i in self.test_data:
            #通过决策树得到预测结果
            if i[-1] == self.__pruned_predict(i,node):
                Sum+=1
        return Sum/len(self.test_data)

    def __pruned_predict(self,x,node):
        '''
        在剪枝树上预测结果
        Params:
            x:预测数据
        '''
        flag = False
        temp_node = self.root
        while temp_node.isLeaf == False:
            if temp_node == node:
                #print('剪枝')
                #该节点为剪枝节点,执行少数服从多数,跳出循环
                return Counter(temp_node.nodeData[:,-1]).most_common(1)[0][0]
            flag = False
            if temp_node.leftChild != None:
                if temp_node.leftChild.isLeaf == True:
                    if x[temp_node.split] == temp_node.leftChild.label:
                        flag = True
                        temp_node = temp_node.leftChild
                elif x[temp_node.split] == temp_node.leftChild.nodeData[0,temp_node.split]:
                    #进到下一个分支
                    flag = True
                    temp_node = temp_node.leftChild
            
            if temp_node.rightChild != None:
                if temp_node.rightChild.isLeaf == True:
                    if x[temp_node.split] == temp_node.rightChild.label:
                        flag = True
                        temp_node = temp_node.rightChild
                elif x[temp_node.split] == temp_node.rightChild.nodeData[0,temp_node.split]:
                    #进到下一个分支
                    flag = True
                    temp_node = temp_node.rightChild                

            if flag == False:
                #如果没匹配到,少数服从多数
                return Counter(temp_node.nodeData[:,-1]).most_common(1)[0][0]
        return temp_node.label

    def buildGraphTree(self,currentNode,G):
        '''
        使用Digraph可视化树,先序遍历
        '''
        if currentNode.leftChild != None or currentNode.rightChild != None:
            #父节点为红色矩形
            G.node(str(currentNode.nodeData),str(len(currentNode.nodeData)),shape='box',color='red')
            if currentNode.leftChild != None:
                G.edge(str(currentNode.nodeData),str(currentNode.leftChild.nodeData),str(columns[currentNode.split])+str(np.unique(currentNode.leftChild.nodeData[:,currentNode.split])))
                if currentNode.leftChild.isLeaf == False:
                    self.buildGraphTree(currentNode.leftChild,G)
            if currentNode.rightChild != None:
                G.edge(str(currentNode.nodeData),str(currentNode.rightChild.nodeData),str(columns[currentNode.split])+str(np.unique(currentNode.rightChild.nodeData[:,currentNode.split])))
                if currentNode.rightChild.isLeaf == False:
                    self.buildGraphTree(currentNode.rightChild,G)
        G.view()
        return 

    def DLR(self,root):
        """
        递归先序遍历
        """
        #print(root.isLeaf,root.nodeData)
        if root.isLeaf == True:
            print("叶子节点","标签",root.label,root.nodeData)
        else:
            print("非叶子节点","分割点",root.split,root.nodeData)
        if root.leftChild != None:
            self.DLR(root.leftChild)
        if root.rightChild != None:
            self.DLR(root.rightChild)
        return        


if __name__ == '__main__':
    '''
    data = loadData('loan_data.csv')
    T = CART_Tree(data)
    T.DLR(T.root)
    '''
    #汽车数据集
    data,columns = loadData('car_evaluate.csv')
    t = CART_Tree(data,columns)
    #t.DLR(t.root)
    g = Digraph("car_evaluate")
    t.buildGraphTree(t.root,g)
    #测试准确率

"""
Author: michealowen
Last edited: 2020.1.02,Thursday
ID3算法生成决策树
"""
#coding=UTF-8

import pandas as pd
import numpy as np
from collections import Counter,defaultdict
from graphviz import Digraph
from sklearn.model_selection import train_test_split

def loadData(fileName):
    """
    载入数据

    Params:
        fileName:数据文件名
    
    Returns:
        data:数组形式数据
    """
    data = pd.read_csv(fileName,delimiter=',',encoding='UTF-8')
    return data.values,data.columns.values

class ID3_Node:
    """ID3树的节点"""

    def __init__(self,isLeaf,childList,nodeData,split,label):
        """
        ID3节点的构造函数

        Params:
            isLeaf:True为叶子节点,False为非叶节点
            childList:该节点的子节点列表
            nodeData:该节点的数据 
            split:在该节点分类的维度(叶节点此属性为-1)
            label:叶子节点拥有类别,非叶子节点为None
        """
        self.isLeaf = isLeaf
        self.childList = childList
        self.nodeData = nodeData
        self.split = split
        self.label = label
        return

class ID3_tree:
    """ID3树类"""

    def __init__(self,data,columns):
        '''
        通过该数据集构造树
        Params:
            data:数据
            columns:属性列表
        '''
        #将数据分为训练集和测试集
        #self.data,self.test_data = train_test_split(data[:,:-1:], data[:,-1], test_size=0.33, random_state=42)[:2:]
        np.random.shuffle(data)
        self.data = data[:int(len(data)*8/10)]
        self.test_data = data[int(len(data)*8/10):]
        print(self.test_data[:,-1])
        self.columns = columns
        if len(self.data) == 0:
            print("数据为空")
            return None 
        self.root = self.bulidTree(self.data,[i for i in range(len(self.data[0])-1)])
        self.REP_pruning()
        #self.Pre_trace(self.root)
        return
    
    def Pre_trace(self,root):
        '''
        先序遍历树
        '''
        if root == None:
            return
        print(len(root.nodeData),'\n')
        if root.isLeaf == True:
            return
        else:
            for node in root.childList:
                self.Pre_trace(node)

    def buildMap(self,data):
        '''
        通过data创建字典
        Params:
            data:数据
        Returns:
            d:Key为label,Value为label出现的次数
        '''
        d = defaultdict(int)
        for i in range(len(data)):
            d[data[i]] +=1
        return d

    def classifyData(self,data,index):
        '''
        将data按照该维度分类
        Params:
            index:划分维度
        Returns:
            d:字典形式,每一个元素都是二维的numpy数组
        '''
        d = {}
        for i in range(len(data)):
            if data[i,index] not in d.keys():
                d[data[i,index]] = np.array([data[i]])
                #print(d[data[i,index]])
            else:
                d[data[i,index]] = np.vstack((d[data[i,index]],data[i]))         
        return d

    def findMaxKey(self,mapp):
        '''
        寻找mapp中,值最大的键
        '''
        maxValue = 0
        maxKey = -1
        for i in mapp.items():
            if i[1] > maxValue:
                maxValue = i[1]
                maxKey = i[0]
        return maxKey

    def bulidTree(self,data,Artributes):
        '''
        递归地构造树
        Params:
            data:进行构造的数据
            Artributes:可选属性的下标
        '''
        #dim = len(data[0])
        #print(len(data))
        #如果所有样本都是同一类,则返回叶子节点
        #print('特征数:',len(Artributes))
        if len(np.unique(data[:,-1])) == 1: 
            return ID3_Node(True,None,data,-1,data[0,-1]) 
        elif len(Artributes) == 0:
            #如果该节点的可分割维度为空,则返回叶子节点,label为data中最多的
            return ID3_Node(True,None,data,-1,Counter(data[:,-1]).most_common(1)[0][0])

        #选取信息增益最大的维度,并根据该维度的取值将data分为若干子节点,选出维度
        entropy = self.getEntropy(data)
        print("熵为",entropy)
        maxGainValue = 0
        maxGainIndex = -1
        temp = 0
        for i in Artributes:
            #选取信息增益最大的一个维度
            temp = self.getConditionalEntropy(data,i)
            print("条件熵为",temp)
            if (entropy - temp) > maxGainValue:
                maxGainIndex = i
                maxGainValue = entropy - temp
        #按照维度分类
        #print("分类维度",self.columns[maxGainIndex])
        #print(maxGainIndex,Artributes)
        if maxGainIndex == -1:
            #当条件熵都不小于经验熵时,停止分割
            return ID3_Node(True,None,data,-1,Counter(data[:,-1]).most_common(1)[0][0])
        Artributes.remove(maxGainIndex)
        nodeList = []
        for element in self.classifyData(data,maxGainIndex).values():
            #print(len(element))
            nodeList.append(self.bulidTree(element,Artributes.copy())) #注意要用深拷贝

        return ID3_Node(False,nodeList,data,maxGainIndex,None) 

    def getEntropy(self,data):
        '''
        求出熵
        Params:
            data:数据集,最后一维参与运算
        Returns:
            entropy:熵
        '''
        if len(data) == 0:
            print("数据为空")
            return
        #dim = len(data[0])
        mapp = defaultdict(int)
        for i in range(len(data)):
            mapp[data[i][-1]] +=1
        entropy = 0
        for i in mapp.keys():
            entropy -= mapp[i]/len(data)*np.log2(mapp[i]/len(data))
        return entropy

    def getConditionalEntropy(self,data,index):
        '''
        求出条件熵        
        Params:
            data:数据集
            index:条件的维度
        Returns:
            sumConditionEntrop:条件熵
        '''
        if len(data) == 0:
            print("数据为空")
            return
        
        #dim = len(data[0])
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
            mapp = defaultdict(int)
            for j in element:
                mapp[data[j][-1]] += 1
            
            for i in mapp.keys():
                sumConditionEntrop -= mapp[i]/len(element)*np.log2(mapp[i]/len(element))*len(element)/len(data)
        #print(indexCategory)
        #print(sumConditionEntrop)
        return sumConditionEntrop

    def REP_pruning(self):
        '''
        后剪枝算法,使用准确率进行剪枝,Reduced-Error Pruning(REP,错误率降低剪枝）
        bottom-up,自底而上进行遍历
        '''
        #使用list来存储树结构,进行自底向上的遍历    首先将节点都入栈中
        node_stack = [self.root]
        cursor = 0
        while True:
            if node_stack[cursor].isLeaf == False:
                for i in node_stack[cursor].childList:
                    node_stack.append(i)
            cursor+=1
            if cursor==len(node_stack):
                break
        cursor-=1
        print('剪枝前节点数',len(node_stack),cursor)
        #不断出栈,进行剪枝,此时游标指向栈顶
        while cursor >= 0:
            #不断向上寻找非叶节点
            if node_stack[cursor].childList==None:
                cursor-=1
            else:
                #找到非叶节点后,若在该节点剪枝后,准确率的到提升,则进行剪枝,使改节点成为叶节点
                p = self.evaluate()
                q = self.__pruned_evaluate(node_stack[cursor])
                #if self.evaluate() <= self.__pruned_evaluate(node_stack[cursor]):
                if p < q:
                    print('准确率提升:',p,'->',q)
                    #递归的删除该节点的子树
                    self.__del_tree(node_stack,node_stack[cursor])
                    '''
                    for i in node_stack[cursor].childList:
                        node_stack.remove(i)
                    node_stack[cursor].childList = None
                    node_stack[cursor].isLeaf = True
                    node_stack[cursor].label = Counter(node_stack[cursor].nodeData[:,-1]).most_common(1)[0][0]
                    node_stack[cursor].split =-1
                    '''
                cursor-=1
        print('剪枝后节点数',len(node_stack))
        return None

    def __del_tree(self,node_stack,node):
        if node.childList == None:
            node_stack.remove(node)
        else:
            for i in node.childList:
                self.__del_tree(node_stack,i)
            #将其变为叶子节点
            node.childList == None
            node.isLeaf = True
            node.label = Counter(node.nodeData[:,-1]).most_common(1)[0][0]
            node.split =-1
        return 


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
        在剪枝树上预测结果,若查询路径上有剪枝节点,则在该节点处停止查询
        Params:
            x:预测数据
        '''
        flag = False        #flag 表示在每一层的搜索中,是否找到应分配到的位置
        temp_node = self.root
        while temp_node.isLeaf == False:
            if temp_node == node:
                #print('yes')
                #该节点为剪枝节点,执行少数服从多数,跳出循环
                return Counter(temp_node.nodeData[:,-1]).most_common(1)[0][0]
            flag = False
            for j in temp_node.childList:
                if j.isLeaf == True:
                    if x[temp_node.split] == j.label:
                        flag = True
                        temp_node = j
                        print('匹配到')
                        break
                elif x[temp_node.split] == j.nodeData[0,temp_node.split]:
                    #进到下一个分支
                    flag = True
                    temp_node = j
                    #print('匹配到')
                    break
            if flag == False:
                #如果没匹配到,少数服从多数
                #print('没匹配到')
                return Counter(temp_node.nodeData[:,-1]).most_common(1)[0][0]
        return temp_node.label

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
            for j in temp_node.childList:
                if j.isLeaf == True:
                    if x[temp_node.split] == j.label:
                        flag = True
                        temp_node = j
                        break
                elif x[temp_node.split] == j.nodeData[0,temp_node.split]:
                    #进到下一个分支
                    flag = True
                    temp_node = j
                    break
            if flag == False:
                #如果没匹配到,少数服从多数
                return Counter(temp_node.nodeData[:,-1]).most_common(1)[0][0]
        return temp_node.label
    
    def buildGraphTree(self,currentNode,G):
        '''
        使用Digraph可视化树,先序遍历
        '''
        if currentNode.childList != None:
            #父节点为红色矩形
            G.node(str(currentNode.nodeData),str(len(currentNode.nodeData)),shape='box',color='red')
            for node in currentNode.childList:
                if node != None:
                    G.edge(str(currentNode.nodeData),str(node.nodeData),str(columns[currentNode.split])+str(np.unique(node.nodeData[:,currentNode.split])))
                    if node.isLeaf == False:
                        self.buildGraphTree(node,G)
        G.view()
        return 

if __name__ == '__main__':
    
    '''
    #贷款数据集
    data,columns = loadData('loan_data.csv')
    t = ID3_tree(data,columns)
    #t.Pre_trace(t.root)
    g = Digraph("loan")
    t.buildGraphTree(t.root,g)
    #测试准确率
    print(t.evaluate())
    '''

    '''
    #乳腺癌数据集
    data,columns = loadData('breast_canser.csv')
    t = ID3_tree(data,columns)
    g = Digraph('breast_canser')
    t.buildGraphTree(t.root,g)
    #测试准确率
    print(t.evaluate())   
    '''

    '''
    #西瓜数据集
    data,columns = loadData('watermelon.csv')
    t = ID3_tree(data,columns)
    #测试准确率
    print(t.evaluate())  
    g = Digraph("watermelon")
    t.buildGraphTree(t.root,g)
    '''

    
    #汽车数据集
    data,columns = loadData('car_evaluate.csv')
    t = ID3_tree(data,columns)
    #测试准确率
    print(t.evaluate())  
    g = Digraph("car_evaluate")
    t.buildGraphTree(t.root,g)
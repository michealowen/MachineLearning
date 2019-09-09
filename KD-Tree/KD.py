"""
Author: michealowen
Last edited: 2019.7.15,Monday
实现KNN,KD树算法
"""
#coding=UTF-8

import operator as op
import numpy as np

#定义两个全局变量
nearestNode = None
nearestValue = float('inf')

class KD_Node:
    """KD树节点类"""
    def __init__(self,point,spilt,leftChild=None,rightChild=None,visited=None):
        """
        KD_Node的构造方法
        
        Args:
            point:节点的向量
            leftChild：左孩子向量
            rightChild：右孩子向量
            spilt：该处的划分维度
            visited:该节点是否访问过
        """
        self.point = point
        self.spilt = spilt
        self.leftChild = None
        self.rightChild = None
        self.visited = False
        return

class KD_Tree:
    """KD树类"""
    def __init__(self,l):
        """
        树的构造方法
        Args:
            l:构造树的列表
        """
        l.sort(key=op.itemgetter(0))
        self.root = KD_Node(l[int(len(l)/2)],0)
        self.bulidTree(l,self.root,int(len(l)/2))
        return

    def bulidTree(self,l,root,index):
        """
        递归的建立树

        Args:
            l:当前进行构建树的列表
            root:当前树根节点
            index:root在l的下标
        """
        if len(l)<=1:
            return
        dim = len(root.point)
        spl = (root.spilt+1)%dim
        #将左右两侧列表排序
        l1,l2 = self.partSort(l,index,spl)
        if len(l1)!=0:
            root.leftChild = KD_Node(l1[int(len(l1)/2)],spl)
            self.bulidTree(l1,root.leftChild,int(len(l1)/2))
        else:
            return
        if len(l2)!=0:
            root.rightChild = KD_Node(l2[int(len(l2)/2)],spl)
            self.bulidTree(l2,root.rightChild,int(len(l2)/2))
        else:
            return 

    def partSort(self,l,index,dim):
        """
        对[0:index]和[index+1:]元素进行排序，排序标准为dim

        Args: 
            index:根节点在当前list的下标

        Returns:
            leftList:已排序的左子列表
            rightList:已排序的右子列表
        """
        leftList = sorted(l[0:index],key=op.itemgetter(dim))
        rightList = sorted(l[index+1:],key=op.itemgetter(dim))
        return leftList,rightList
    
    def LDR(self,root):
        """
        中序遍历
            1.中序遍历左子树
            2.访问根节点
            3.中序遍历右子树
        
        Args:
            root:树的根节点
        """
        if root.leftChild!=None:
            self.LDR(root.leftChild)
        print(root.point)
        if root.rightChild!=None:
            self.LDR(root.rightChild)
        return

    def DLR(self,root):
        """
        先序遍历
            1.访问根节点
            2.先序遍历左子树
            3.先序遍历右子树
        Args:
            root:树的根节点
        """
        print(root.point)
        if root.leftChild!=None:
            self.DLR(root.leftChild)
        if root.rightChild!=None:
            self.DLR(root.rightChild)
        return
    
    def LRD(self,root):
        """
        后序遍历
            1.后序遍历左子树
            2.后序遍历右子树
            3.访问根节点
        Args:
            root:树的根节点
        """
        if root.leftChild!=None:
            self.LRD(root.leftChild)
        if root.rightChild!=None:
            self.LRD(root.rightChild)
        print(root.point)
        return

    def EcludDis(self,s,t):
        """
        计算欧几里得距离

        Args:
            s:参与计算的向量s
            t:参与计算的向量t

        Returns:欧几里得距离

        """
        return sum((np.array(s)-np.array(t))**2)

    def nearestNeighbor(self,point,currentNode):
        """
        寻找point的最近邻居
        
        Args:
            point:要寻找的点
            currenNode:进行寻找的树的根
        """
        global nearestNode,nearestValue
        if currentNode == None:
            return 
        currentNode.visited = True

        #index为当前的区分维度
        index=currentNode.spilt

        if point[index] < currentNode.point[index]:
            self.nearestNeighbor(point,currentNode.leftChild)
        if point[index] > currentNode.point[index]:
            self.nearestNeighbor(point,currentNode.rightChild)
        dis=self.EcludDis(point,currentNode.point)

        print(dis)
        
        if dis<nearestValue:
            nearestNode=currentNode
            nearestValue=dis
        
        #判断当前节点是否为叶子节点
        if currentNode.rightChild != None or currentNode.leftChild != None:
            #判断分界线与点的垂直距离是否小于最小距离，去未访问过的一边访问
            if abs( point[index] - currentNode.point[index]) <= nearestValue:
                if currentNode.rightChild.visited == False:
                    self.nearestNeighbor(point,currentNode.rightChild)
                elif currentNode.leftChild.visited == False:
                    self.nearestNeighbor(point,currentNode.leftChild)
        
        return

    def main(self):
        """
        测试函数
        """
        #首先选取第一个根节点
        #l = [[3,2],[1,7],[4,5],[6,2],[5,7],[2,9],[3,8],[7,4]]
        l = [[2,3],[5,4],[9,6],[6,4.5],[8,1],[7,2]]
        l.sort(key=op.itemgetter(0))
        root = KD_Node(l[int(len(l)/2)],0)
        self.bulidTree(l,root,int(len(l)/2))
        #self.LDR(root)
        #self.DLR(root)
        self.nearestNeighbor([6,3.5],root)
        print(nearestNode.point)
        print(nearestValue)
        return

l = [[2,3],[5,4],[9,6],[6,4.5],[8,1],[7,2]]
t = KD_Tree(l)
t.nearestNeighbor([6,3.5],t.root)
#t.partSort([[3,2],[1,7],[4,5],[6,2],[5,7]],2,0)

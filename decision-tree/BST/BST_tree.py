"""
Author: michealowen
Last edited: 2019.8.6,Thursday
生成二叉排序树
"""
#encoding=UTF-8

import numpy as np
#from graphviz import Digraph
import networkx as nx
import matplotlib.pyplot as plt
from graphviz import Digraph

class node:
    """
    树的节点
    """

    def __init__(self,data,leftChild,rightChild):
        self.data = data
        self.leftChild = leftChild
        self.rightChild = rightChild
        return

class BST_Tree:
    """
    排序二叉树
    """

    def __init__(self,l):
        """
        二叉树的构造方法
        """
        self.l = l
        self.root = self.buildTree(np.sort(l))
        return

    def buildTree(self,l):
        """
        递归地构建树
        """
        if len(l) == 0:
            return None
        #首先找出根节点
        root = node(l[int(len(l)/2)],self.buildTree(l[0:int(len(l)/2)]),self.buildTree(l[int(len(l)/2)+1:]))
        return root

    def LDR(self,root):
        """
        中序遍历树
        """
        if root == None:
            return
        self.LDR(root.leftChild)
        print(root.data)
        self.LDR(root.rightChild)
        return

    def vizTree(self,root,G):
        """
        使用plt可视化树(树的形状不够清楚)
        """
        G.add_node(root.data)
        if root.leftChild != None:
            G.add_edge(root.data, root.leftChild.data)
            self.vizTree(root.leftChild,G)
        if root.rightChild != None:
            G.add_edge(root.data, root.rightChild.data)
            self.vizTree(root.rightChild,G)
        return

    def buildGraphTree(self,currentNode,G):
        """
        使用Digraph可视化树,先序遍历
        """

        G.node(str(currentNode.data),str(currentNode.data))
        if currentNode.leftChild != None:
            G.edge(str(currentNode.data),str(currentNode.leftChild.data))
            self.buildGraphTree(currentNode.leftChild,G)
        if currentNode.rightChild != None:
            G.edge(str(currentNode.data),str(currentNode.rightChild.data))
            self.buildGraphTree(currentNode.rightChild,G)
        #G.view()
        return None
    
        
t = BST_Tree([3,6,1,7,8,5,0,4,2,9])
t.LDR(t.root)

'''
使用plt的方式
G = nx.DiGraph()
t.vizTree(t.root,G)
nx.draw(G,with_labels=True)
plt.show()
'''

g = Digraph("G")
t.buildGraphTree(t.root,g)
print(g.body)
g.view()

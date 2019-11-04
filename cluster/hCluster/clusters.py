"""
Author: Michealowen
Last edited:2019.9.10,Tuesday
层次聚类HierarchyCluster
"""
#encoding=UTF-8

import numpy as np
from PIL import Image,ImageDraw,ImageFont 

class bicluster:
    def __init__(self,vec,Id,left=None,right=None,distance=0.0):
        """
        Args:
            distance:左右子簇的距离
        """
        self.vec=vec
        self.Id=Id
        self.left=left
        self.right=right
        self.distance=distance

def readFile(filename):
    """
    加载数据文件
    Args:
        filename:文件名
    Returns:
        rowNames:行名(博客名)
        colNames:列名(关键词)
        data:数据矩阵
    """
    lines = np.array([line for line in open(filename)])
    #第一行是列标题
    colNames=lines[0].strip().split('\t')[1:]
    rowNames=np.array([])
    data=[]
    for line in lines[1:]:
        #每行的第一列为行名
        p=line.strip().split('\t')
        rowNames=np.append(rowNames,p[0])
        data.append([float(x) for x in p[1:]])
    return rowNames,colNames,np.array(data)

def pearson(v1,v2):
    """
    计算两个向量的皮尔森相似度
    Args:
        v1:第一个向量
        v2:第二个向量
    Returns:
        corelation:皮尔森相似度(相关系数)
    """
    v1 = v1/1.0
    v2 = v2/1.0
    cov = np.mean(v1*v2) - np.mean(v1)*np.mean(v2) 
    #cov为协方差
    corelation = cov/(np.sqrt(np.var(v1))*np.sqrt(np.var(v2)))
    #corelationq为相关系数
    return corelation
        
def hcluster(data,distance=pearson):
    """
    构造聚类树
    Args:
        data:数据矩阵
        distance:距离计算标准,此处默认为皮尔森距离
    """
    data=data/1.0
    #在最开始所有的行都为不同的簇
    clusters=np.array([bicluster(data[i],i) for i in range(len(data))])
    #print(clusters[16])

    #dis为距离表
    dis={}
    #直至簇数量收缩为1结束循环
    currentID=0
    while len(clusters)>1:
        minDis=2 #因为最大距离不会超过2
        #每次寻找出距离最近的两个簇
        for i in range(len(clusters)):
            for j in range(i+1,len(clusters)):
                #计算距离
                dis[i,j]=(1-distance(clusters[i].vec,clusters[j].vec))
                if dis[i,j]<minDis:
                    minDis=dis[i,j]
                    nearestPair=(i,j)
        
        #找出最近簇之后,用平均值生成新簇,覆盖原簇
        newDis=minDis
        newVec=(clusters[nearestPair[0]].vec+clusters[nearestPair[1]].vec)/2
        currentID-=1

        print(clusters[nearestPair[0]].Id,clusters[nearestPair[1]].Id)
        newCluster=bicluster(newVec,currentID,clusters[nearestPair[0]],clusters[nearestPair[1]],newDis)
        print(newCluster.left.Id,newCluster.right.Id)
        clusters=np.delete(clusters,[nearestPair[0],nearestPair[1]],0)
        clusters=np.append(clusters,np.array([newCluster]),0)
    return clusters[0]

def LDR(root):
    """
    中序遍历树
    """
    if root == None:
        return
    LDR(root.left)
    print(root.Id)
    LDR(root.right)
    return


def getheight(clust):  
    # Is this an endpoint? Then the height is just 1  
    if clust.left==None and clust.right==None: return 1  
  
    # Otherwise the height is the same of the heights of  
    # each branch  
    return getheight(clust.left)+getheight(clust.right)  
#计算误差  
def getdepth(clust):  
    # The distance of an endpoint is 0.0  
    if clust.left==None and clust.right==None: return 0  
  
    #一个枝节点的距离等于左右两侧分支中距离较大者加上该枝节点自身的距离  
    return max(getdepth(clust.left),getdepth(clust.right))+clust.distance  
  
#画图  
def drawdendrogram(clust,labels,jpeg='clusters.jpg'):  
    # height and width  
    h=getheight(clust)*30  
    w=1200  
    depth=getdepth(clust)  
  
    # width is fixed, so scale distances accordingly  
    scaling=float(w-350)/depth  
  
    # Create a new image with a white background  
    img=Image.new('RGB',(w,h),(255,255,255))  
    draw=ImageDraw.Draw(img)  
  
    draw.line((0,h/2,10,h/2),fill=(255,0,0))      
  
    # Draw the first node  
    drawnode(draw,clust,10,(h/2),scaling,labels)  
    img.save(jpeg,'JPEG')  
  
def drawnode(draw,clust,x,y,scaling,labels):  
    if clust.Id<0:  
        h1=getheight(clust.left)*20  
        h2=getheight(clust.right)*20  
        top=y-(h1+h2)/2  
        bottom=y+(h1+h2)/2  
        # Line length  
        ll=clust.distance*scaling  
        # Vertical line from this cluster to children      
        draw.line((x,top+h1/2,x,bottom-h2/2),fill=(255,0,0))      
      
        # Horizontal line to left item  
        draw.line((x,top+h1/2,x+ll,top+h1/2),fill=(255,0,0))      
  
        # Horizontal line to right item  
        draw.line((x,bottom-h2/2,x+ll,bottom-h2/2),fill=(255,0,0))          
  
        # Call the function to draw the left and right nodes      
        drawnode(draw,clust.left,x+ll,top+h1/2,scaling,labels)  
        drawnode(draw,clust.right,x+ll,bottom-h2/2,scaling,labels)  
    else:     
        # If this is an endpoint, draw the item label  
        print(type(labels[clust.Id]))
        #此处一定要使用系统中有的字体
        font = ImageFont.truetype('LiberationMono-BoldItalic.ttf',24) 
        draw.text((x+5,y-7),labels[clust.Id],(0,0,0),font) 

x = pearson(np.array([1,2,3,4]),np.array([2,4,6,8]))
print(x)
#print(readFile('blogdata1.txt')[2])
blognames,keywords,data=readFile('blogdata1.txt')
temp=hcluster(data)
#print(temp.left.id)
LDR(temp)
drawdendrogram(temp,blognames,'blogclust.jpg')
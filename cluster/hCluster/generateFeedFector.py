"""
Author: Michealowen
Last edited:2019.9.10,Tuesday
博客单词获取
"""
#encoding=UTF-8

import feedparser
import re
import jieba  

def getWordsCounts(url):
    '''
    Args:
        解析的链接
    Return:
        RSS订阅源的标题和包含单词计数情况的字典
    '''
    # 解析订阅源
    d = feedparser.parse(url)
    wc = {}

    # 循环遍历所有的文章条目
    for e in d.entries:
        if 'summary' in e:
            summary = e.summary
        else:
            summary = e.description

        # 提取一个单词列表
        words=getwords(e.title+' '+summary)
        for word in words:
            wc.setdefault(word,0)
            wc[word]+=1
    return d.feed.title,wc

def getwords(html):
    # 去除所有HTML标记
    txt=re.compile(r'<[^>]+>').sub('',html)
    algor = jieba.cut(txt,cut_all=True)  
    # 利用所有非字母字符拆分出单词
    #words=re.compile(r'[^A-Z^a-z]+').spilt(txt)

    return [tok.lower() for tok in algor if tok!=''] 
    #转换成小写形式
    #return [word.lower() for word in words if word!='']

apcount={}  
wordcounts={}  
feedlist=[line for line in open('feedlist.txt')]  
for feedurl in feedlist:    
    title,wc=getWordsCounts(feedurl)  
    wordcounts[title]=wc  
    for word,count in wc.items():  
        apcount.setdefault(word,0)  
        if count>1:  
            apcount[word]+=1   
  
wordlist=[]  
for w,bc in apcount.items():  
    frac=float(bc)/len(feedlist)  
    if frac>0.1 and frac<0.5:       
    #因为像“我”这样的单词几乎到处都是，而像“PYTHON”这样的单词则有可能只出现在个别博客中，所以通过只选择介于某个百分比范围内的单词，我们可以减少需要考查的单词总量。我们这里将1/10为下界，5/10为上界。  
        wordlist.append(w)  
  
out=open('blogdata1.txt','w')  
out.write('Blog')  
for word in wordlist: 
    out.write('\t%s' % word.strip())  
out.write('\n')  
for blog,wc in wordcounts.items():  
    out.write(blog)  
    for word in wordlist:  
        if word in wc: out.write('\t%d' % wc[word])  
        else: out.write('\t0')  
    out.write('\n')  
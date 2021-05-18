# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 11:07:29 2021

@author: Caoer
"""

'''
eg.Python流式读取MYSQL数据的参考代码
'''
import sqlalchemy
def sql_data_generator():
    db = sqlalchemy.create_engine('mysql+pymysql://user:password@123.456.789.123/yourdatabase?charset=utf-8')
    result = db.execution_options(stream_results=True).execute(sqlalchemy.text('select content from articles'))
    for t in result:
        yield t[0]


'''
eg.python流式读取Mongo数据的参考代码
'''
import pymongo

db = pymongo.MongoClient().baike.items
def texts():
    for a in db.find(no_cursor_timeout=True).limit(1000000):
        yield a['content']


'''
【算法思路】：只用频数和凝固度，去掉了计算量最大的边界熵，计算凝固度时只计算二字片段的凝固度，能够得到任意长度的词语。
'''

from collections import defaultdict # defaultdict是经过封装的dict，它能够让我们设定默认值
from tqdm import tqdm # tqdm是一个非常易用的用来显示进度的库
import re


class Find_Words:
    def __init__(self, min_count=10, min_pmi=0):
        """
        :param min_count:最小出现的频次
        :param min_pmi:最小凝固度阈值
        """
        self.min_count = min_count
        self.min_pmi = min_pmi
        # 如果键不存在，那么就用int函数，初始化一个值，int()的默认结果为0
        self.chars, self.pairs = defaultdict(int), defaultdict(int)
        
        self.total = 0
    
    def text_filter(self, texts):
        """
        预切断句子，以免得到太多无意义（不是中文、英文、数字）的字符串
        :param texts:文档
        :return:
        """
        for a in tqdm(texts):
            # for a in texts:
            # 这个正则表达式匹配的是任意非中文、非英文、非数字，因此它的意思就是用任意非中文、非英文、非数字的字符断开句子
            for t in re.split(u'[^\u4e00-\u9fa50-9a-zA-Z]+', a):
                if t is not None:
                    yield t
    
    def count(self, texts):
        """
        计数函数，计算单字出现频数、相邻两字出现频数
        :param texts:
        :return:
        """
        for text in self.text_filter(texts):
            self.chars[text[0]] += 1
            for i in range(len(text)-1):
                # 统计unigram出现次数
                self.chars[text[i+1]] += 1
                # 统计前序bigram的出现次数
                self.pairs[text[i:i+2]] += 1
                # 统计全局词数量
                self.total += 1
        # 过滤掉频率小于min_count的
        self.chars = {i: j for i, j in self.chars.items() if j >= self.min_count} # 最少频数过滤
        self.pairs = {i: j for i, j in self.pairs.items() if j >= self.min_count} # 最少频数过滤
        # 计算凝固度
        self.strong_segments = {i: self.total * j / (self.chars[i[0]] * self.chars[i[1]]) for i, j in
                                self.pairs.items()}
        # 过滤掉概率小于阈值1的，即那些不能成词的片段
        self.strong_segments = {i: j for i, j in self.strong_segments.items() if j >= self.min_pmi}
        
        # self.strong_segments = set()
        # for i,j in self.pairs.items(): # 根据互信息找出比较“密切”的邻字
        #     _ = log(self.total*j/(self.chars[i[0]]*self.chars[i[1]]))
        #     if _ >= self.min_pmi:
        #         self.strong_segments.add(i)
    
    def find_words(self):  # 根据前述结果来找词语
        self.words = defaultdict(int)
        self.total_words = 0
        for text in self.text_filter(texts):
            s = text[0]
            for i in range(len(text)-1):
                # 如果比较“密切”则不断开，即text[i]+text[i+1]在分词词典中，那么s += text[i+1]
                if text[i:i+2] in self.strong_segments:
                    s += text[i+1]
                else:
                    self.words[s] += 1  # 否则断开，前述片段作为一个词来统计
                    self.total_words += 1
                    s = text[i+1]
            self.words[s] += 1  # 最后一个“词”
        self.words = {i: j for i, j in self.words.items() if j >= self.min_count} # 最后再次根据频数过滤


if __name__ == "__main__":
    # texts = ['中国首都是北京，北京市著名景点有颐和园', '北京市是中国的首都']
    # fw = Find_Words(2, 1)
    # fw.count(texts)
    # fw.find_words()
    #
    # import pandas as pd
    # words = pd.Series({k: v for k, v in fw.words.items() if len(k) >= 2}).sort_values(ascending=False)
    # print(words)
    
    texts = []
    fw = Find_Words(30, 50)
    fileName = '天龙八部（世纪新修版）'
    f = open(fileName+'.txt', 'r')
    texts.append(f.read()[:1000000])  # 读取为一个字符串
    drop_dict = [u' ', u'\u3000', u'/']
    for i in drop_dict:  # 去掉标点字
        texts[0] = texts[0].replace(i, '')
    fw.count(texts)
    fw.find_words()

    import pandas as pd
    words = pd.Series({k: v for k, v in fw.words.items() if len(k) >= 2}).sort_values(ascending=False)
    print(words)
    words.to_csv(fileName+'_result.txt', header=False)

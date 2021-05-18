"""
 @Description: 主要用来参加Data Fountain举办的“电力专业领域词汇挖掘”比赛
 			首先pip install nlp-zero==0.1.6,整个库纯python实现，没有第三方调用，支持python2.x和3.x。
			参考链接：https://kexue.fm/archives/5597
 @Author: XjCao
 @Date 2021/3/23 14:43
"""
from nlp_zero import *
import re
import pandas as pd
import pymongo
import numpy as np
from multiprocessing.dummy import Queue
import logging
from gensim.models import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')


class D:  # 读取比赛方所给语料
	def __iter__(self):
		with open('data.txt', encoding='utf-8') as f:
			for l in f:
				l = l.strip()
				l = re.sub(u'[^\u4e00-\u9fa5]+', ' ', l)
				yield l


class DO:  # 读取自己的语料（相当于平行语料）
	def __iter__(self):
		with open('data.txt', encoding='utf-8') as f:
			for l in f:
				l = l.strip()
				l = re.sub(u'[^\u4e00-\u9fa5]+', ' ', l)
				yield l


class DW:
	def __iter__(self):
		for l in D():
			yield tokenizer.tokenize(l, combine_Aa123=False)


def is_good(w):
	"""
	进行规则过滤，由于即使进行语义聚类，仍旧会出现非电力词汇例如麦克斯韦方程，甚至保留了一些“非词”
	:param w:
	:return:
	"""
	if re.findall(u'[\u4e00-\u9fa5]', w) \
			and len(w) >= 2 \
			and not re.findall(u'[较很越增]|[多少大小长短高低好差]', w) \
			and not u'的' in w \
			and not u'了' in w \
			and not u'这' in w \
			and not u'那' in w \
			and not u'到' in w \
			and not w[-1] in u'为一人给内中后省市局院上所在有与及厂稿下厅部商者从奖出' \
			and not w[0] in u'每各该个被其从与及当为' \
			and not w[-2:] in [u'问题', u'市场', u'邮件', u'合约', u'假设', u'编号', u'预算', u'施加', u'战略', u'状况', u'工作', u'考核',
							   u'评估', u'需求', u'沟通', u'阶段', u'账号', u'意识', u'价值', u'事故', u'竞争', u'交易', u'趋势', u'主任',
							   u'价格', u'门户', u'治区', u'培养', u'职责', u'社会', u'主义', u'办法', u'干部', u'员会', u'商务', u'发展',
							   u'原因', u'情况', u'国家', u'园区', u'伙伴', u'对手', u'目标', u'委员', u'人员', u'如下', u'况下', u'见图',
							   u'全国', u'创新', u'共享', u'资讯', u'队伍', u'农村', u'贡献', u'争力', u'地区', u'客户', u'领域', u'查询',
							   u'应用', u'可以', u'运营', u'成员', u'书记', u'附近', u'结果', u'经理', u'学位', u'经营', u'思想', u'监管',
							   u'能力', u'责任', u'意见', u'精神', u'讲话', u'营销', u'业务', u'总裁', u'见表', u'电力', u'主编', u'作者',
							   u'专辑', u'学报', u'创建', u'支持', u'资助', u'规划', u'计划', u'资金', u'代表', u'部门', u'版社', u'表明',
							   u'证明', u'专家', u'教授', u'教师', u'基金', u'如图', u'位于', u'从事', u'公司', u'企业', u'专业', u'思路',
							   u'集团', u'建设', u'管理', u'水平', u'领导', u'体系', u'政务', u'单位', u'部分', u'董事', u'院士', u'经济',
							   u'意义', u'内部', u'项目', u'建设', u'服务', u'总部', u'管理', u'讨论', u'改进', u'文献'] \
			and not w[:2] in [u'考虑', u'图中', u'每个', u'出席', u'一个', u'随着', u'不会', u'本次', u'产生', u'查询', u'是否', u'作者'] \
			and not (u'博士' in w or u'硕士' in w or u'研究生' in w) \
			and not (len(set(w)) == 1 and len(w) > 1) \
			and not (w[0] in u'一二三四五六七八九十' and len(w) == 2) \
			and re.findall(u'[^一七厂月二夕气产兰丫田洲户尹尸甲乙日卜几口工旧门目曰石闷匕勺]', w) \
			and not u'进一步' in w:
		return True
	else:
		return False


def most_similar(word, center_vec=None, neg_vec=None):
	"""
	根据给定词、中心向量和负向量找最近的词
	:param word:
	:param center_vec:
	:param neg_vec:
	:return:
	"""
	vec = word2vec[word] + center_vec - neg_vec
	return word2vec.similar_by_word(vec, topn=20)


def find_words(start_words, center_words=None, neg_words=None, min_sim=0.6, max_sim=1., alpha=0.25):
	if center_words == None and neg_words == None:
		min_sim = max(min_sim, 0.6)
	center_vec, neg_vec = np.zeros([word_size]), np.zeros([word_size])
	if center_words:  # 中心向量是所有中心种子词向量的平均
		_ = 0
		for w in center_words:
			if w in word2vec.wv.vocab:
				center_vec += word2vec[w]
				_ += 1
		if _ > 0:
			center_vec /= _
	if neg_words:  # 负向量是左右负种子词向量的平均（此处无用）
		_ = 0
		for w in neg_words:
			if w in word2vec.wv.vocab:
				neg_vec += word2vec[w]
				_ += 1
		if _ > 0:
			neg_vec /= _
	queue_count = 1
	task_count = 0
	cluster = []
	queue = Queue()  # 建立队列
	for w in start_words:
		queue.put((0, w))
		if w not in cluster:
			cluster.append(w)
	while not queue.empty():
		idx, word = queue.get()
		queue_count -= 1
		task_count += 1
		sims = most_similar(word, center_vec, neg_vec)
		min_sim_ = min_sim + (max_sim - min_sim) * (1 - np.exp(-alpha * idx))
		if task_count % 10 == 0:
			log = '%s in cluster, %s in queue, %s tasks done, %s min_sim' % (
			len(cluster), queue_count, task_count, min_sim_)
			print(log)
		for i, j in sims:
			if j >= min_sim_:
				if i not in cluster and is_good(i):  # is_good是人工写的过滤规则
					queue.put((idx + 1), i)
					if i not in cluster and is_good(i):
						cluster.append(i)
					queue_count += 1
	return cluster


if __name__ == "__main__":
	# 在比赛方语料中做新词发现
	f = Word_Finder(min_proba=1e-6, min_pmi=0.5)
	f.train(D())  # 统计互信息
	f.find(D())  # 构建词库
	
	# 导出词表
	words = pd.Series(f.words).sort_values(ascending=False)
	
	# 在自己的语料上做新词发现
	fo = Word_Finder(min_proba=1e-6, min_pmi=0.5)
	fo.train(DO())  # 统计互信息
	fo.find(DO())  # 构建词库
	
	# 导出词表
	other_words = pd.Series(fo.words).sort_values(ascending=False)
	other_words = other_words / other_words.sum() * words.sum()  # 总词频归一化（这样才便于比较）
	
	"""
	对比两份语料词频，得到特征词
	对比指标是（比赛方语料的词频 + alpha） / （自己语料的词频 + beta）;
	alpha和beta的计算参考自http://www.matrix67.com/blog/archives/5044
	"""
	WORDS = words.copy()
	OTHERS_WORDS = other_words.copy()
	
	total_zeros = (WORDS + OTHERS_WORDS).fillna(0) * 0
	words = WORDS + total_zeros
	other_words = OTHERS_WORDS + total_zeros
	total = words + other_words
	
	alpha = words.sum() / total.sum()
	
	result = (words + total.mean() * alpha) / (total + total.mean())
	result = result.sort_values(ascending=False)
	idxs = [i for i in result.index if len(i) >= 2]  # 排除掉单字词
	idxs = [i for i in idxs if is_good(i)]  # 规则过滤
	
	# 导出csv格式
	pd.Series(idxs[:20000]).to_csv('result_1.csv', encoding='utf-8', header=None, index=None)
	
	"""
	训练Word2Vec模型
	"""
	# nlp zero提供了良好的封装，可以直接导出一个分词器，词表是新词发现得到的词表
	tokenizer = f.export_tokenizer()
	word_size = 100
	word2vec = Word2Vec(DW(), size=word_size, min_count=2, sg=1, negative=10)
	
	"""
	根据Word2Vec得到的词向量来对词进行聚类，非严格意义上的聚类，根据挑出来的若干个种子词，找到一批相似词。算法用相似的传递性（有点类似基于连通性的聚类算法），
	即A和B相似，B和C也相似，那么A、B、C就聚为一类。
	"""
	# 种子词，在第一步得到的词表中的前面部分挑出，无需特别准
	start_words = [u'电网', u'电压', u'直流', u'电力系统', u'变压器', u'电流', u'负荷', u'发电机', u'变电站',
				   u'机组', u'母线', u'电容', u'放电', u'等效', u'节点', u'电机', u'故障', u'输电线路', u'波形',
				   u'电感', u'导线', u'继电', u'输电', u'参数', u'无功', u'线路', u'仿真', u'功率', u'短路',
				   u'控制器', u'谐波', u'励磁', u'电阻', u'模型', u'开关', u'绕组', u'电力', u'电厂', u'算法',
				   u'供电', u'阻抗', u'调度', u'发电', u'场强', u'电源', u'负载', u'扰动', u'储能', u'电弧',
				   u'配电', u'系数', u'雷电', u'输出', u'并联', u'回路', u'滤波器', u'电缆', u'分布式',
				   u'故障诊断', u'充电', u'绝缘', u'接地', u'感应', u'额定', u'高压', u'相位', u'可靠性',
				   u'数学模型', u'接线', u'稳态', u'误差', u'电场强度', u'电容器', u'电场', u'线圈', u'非线性',
				   u'接入', u'模态', u'神经网络', u'频率', u'风速', u'小波', u'补偿', u'电路', u'曲线', u'峰值',
				   u'容量', u'有效性', u'采样', u'信号', u'电极', u'实测', u'变电', u'间隙', u'模块', u'试验',
				   u'滤波', u'量测', u'元件', u'最优', u'损耗', u'特性', u'谐振', u'带电', u'瞬时', u'阻尼',
				   u'转速', u'优化', u'低压', u'系统', u'停电', u'选取', u'传感器', u'耦合', u'振荡', u'线性',
				   u'信息系统', u'矩阵', u'可控', u'脉冲', u'控制', u'套管', u'监控', u'汽轮机', u'击穿', u'延时',
				   u'联络线', u'矢量', u'整流', u'传输', u'检修', u'模拟', u'高频', u'测量', u'样本', u'高级工程师',
				   u'变换', u'试样', u'试验研究', u'平均值', u'向量', u'特征值', u'导体', u'电晕', u'磁通', u'千伏', u'切换', u'响应', u'效率']
	cluster_words = find_words(start_words, min_sim=0.6, alpha=0.35)
	
	result2 = result[cluster_words].sort_values(ascending=False)
	idxs = [i for i in result2.index if is_good(i)]
	
	pd.Series([i for i in idxs if len(i) > 2][:10000]).to_csv('result_1_2.csv', encoding='utf-8', header=None,
															  index=None)

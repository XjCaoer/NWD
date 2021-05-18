# 新词发现算法(New Word Detection)
##【算法来源】
互联网时代的社会语言学：基于SNS的文本数据挖掘，http://www.matrix67.com/blog/archives/5044

##【代码来源及引用】
新词发现的信息熵方法与实现，https://kexue.fm/archives/3491

苏剑林. (Oct. 26, 2015). 《新词发现的信息熵方法与实现 》[Blog post]. Retrieved from https://kexue.fm/archives/3491

##【算法知识】
【未登录词】：没有被收录在分词词表中但必须被切分出来的词，包括各种专有名词（人名、地名、企业名、机构名等）、缩写词、网络新词新增词汇等等。
（参考：张剑锋.规则和统计相结合的中文分词方法研究.[硕士学位论文].太原：山西大学，2008.）
	
【频数】一个文本片段在文本中出现的频率，但是概率怎么计算？？？eg.在2400万字的数据中，“电影”一共出现了2774次，出现的概率约为0.000113？
“院”字则出现了4797次，出现的概率约为0.0001969。
	
【凝固程度】：枚举凝合方式，p(x)表示文本片段x在整个语料中出现的概率，p(电影院) = min(p(电影院)/(p(电)*p(影院)), p(电影院)/(p(电影)*p(院)))
	
【信息熵】反映知道一个事件的结果后平均会给你带来多大的信息量。
	
【自由程度】文本片段的自由运用程度：左邻字信息熵和右邻字信息熵中的较小值。min(H左=H(P1, P2,..., Pn)=-P(xi)logP(xi),
H右=H(P1, P2,..., Pn)=-P(xi)logP(xi))

【算法的想法】：不依赖任何已有的词库，仅仅根据词的共同特征，将一段大规模语料中可能成词的文本片段全部提取出来，不管它是新词还是旧词，
再把所有抽出来的词和已有词库进行比较。	

【参数】·   
		-候选词长度上限d——即把文本中出现过的所有长度不超过d的子串都当作潜在的词（即候选词，设定初值为5，根据需求进行修改）
		-出现频数：设定一个阈值
		-凝固程度：设定一个阈值
		-自由程度：设定一个阈值
		提取出所有满足阈值要求的候选词即可。
		经验值：30M文本，词频>200, 凝固度>10**(n-1), 自由度>1.5
		小窍门：词频>30, 凝固度>20**(n-1)也能发现很多低频的词汇。
		
		为了提高效率，把语料全文视作一整个字符串，并对该字符串的所有后缀按字典序排序，在内存中存储这些后缀的前d+1个字，或者只存储它们在语料中的起始位置。
		
		从头到尾扫描一遍算出各个候选词的频数和右邻字信息熵。
		将整个语料逆序后重新排列所有的后缀，再扫描一遍后统计出每个候选词的左邻字信息熵。
		根据频数信息计算凝固程度。
		效率O(nlogn)
		
		Bayesian average —— 与全局平均取加权平均的思想，最常见的平滑处理方法是分子分母都加上一个常数。
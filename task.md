1、收集语料
开源数据：参考地址：https://huggingface.co/datasets 代理地址：https://hf-mirror.com/datasets
eg:科技与研究 https://huggingface.co/datasets/BAAI/IndustryCorpus2_technology_scientific_research
按不同指令收集，每个指令至少1000条，字数300-2000字

2、数据预处理
整理为Excel，格式如下：
指令|原文|译文

3、谷歌翻译
指令|原文|译文|谷歌译文
可以直接上传excel文件翻译

4、翻译质量评价
指令|原文|译文|谷歌译文|LLM评价

eg:请比较以下两种翻译，并判断哪个翻译更流畅、更准确地表达了原文的意思。
原文：[原文内容]
人工翻译：[人工翻译内容]
谷歌翻译：[谷歌翻译内容]
请给出你的评价理由，并选择更优的翻译。

规则：其中0分表示“没有保留任何意义”，100分表示“完美的意义和语法”


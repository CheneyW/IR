#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/9
"""

import math
import json

PARAM_K1 = 1.5
PARAM_B = 0.75


class BM25(object):

    def __init__(self, corpus, **kwargs):
        if 'path' in kwargs:
            self.load(kwargs['path'])
            return

        self.corpus_size = len(corpus)  # 文档数
        self.doc_len = [len(doc) for doc in corpus]  # 所有文档长度
        self.avg_doc_len = sum(self.doc_len) / self.corpus_size  # 文档平均长度

        self.doc_freqs = []  # 每个文档中每个词出现的次数
        self.idf = {}  # log(idf)

        w2dnum = {}  # word -> doc num  出现每个词的文档数
        for doc in corpus:
            frequencies = {}  # 文档中每个词出现的次数
            for word in doc:
                frequencies[word] = 1 if word not in frequencies else frequencies[word] + 1
            self.doc_freqs.append(frequencies)

            for word, freq in frequencies.items():
                w2dnum[word] = 1 if word not in w2dnum else w2dnum[word] + 1

        for word, freq in w2dnum.items():
            self.idf[word] = math.log(self.corpus_size) - math.log(freq)

    # 搜索词和指定文档之间相关性
    def get_score(self, query, index):
        doc_freqs = self.doc_freqs[index]
        score = sum([self.idf[word] * doc_freqs[word] * (PARAM_K1 + 1) /
                     (doc_freqs[word] + PARAM_K1 * (1 - PARAM_B + PARAM_B * self.doc_len[index] / self.avg_doc_len))
                     for word in query if word in doc_freqs])
        return score

    # 搜索词和所有文档之间相关性
    def get_scores(self, query):
        scores = [self.get_score(query, index) for index in range(self.corpus_size)]
        return scores

    # 保存模型
    def save(self, path):
        this = dict()
        this['corpus_size'] = self.corpus_size
        this['doc_len'] = self.doc_len
        this['avg_doc_len'] = self.avg_doc_len
        this['doc_freqs'] = self.doc_freqs
        this['idf'] = self.idf

        with open(path, 'w', encoding='utf-8')as f:
            json.dump(this, f)

    # 加载模型
    def load(self, path):
        with open(path, 'r', encoding='utf-8')as f:
            this = json.load(f)
        self.corpus_size = this['corpus_size']
        self.doc_len = this['doc_len']
        self.avg_doc_len = this['avg_doc_len']
        self.doc_freqs = this['doc_freqs']
        self.idf = this['idf']

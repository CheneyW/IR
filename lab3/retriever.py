#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/9

从数据中检索数据 & 检索附件
"""

import os
import json
import numpy as np
from bm25 import BM25
from inverted_index import InvertedIndex
from pyltp import Segmentor

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, 'data')
ATTACHMENT_PATH = os.path.join(DATA_PATH, 'attachment')
MODEL_PATH = os.path.join(DIR_PATH, 'model')


class Retriever(object):
    def __init__(self):
        self.data = self._get_data()
        self.idx2file = []
        for idx, data in enumerate(self.data):
            self.idx2file += [(idx, i) for i in range(len(data['segmented_file_name']))]

        corpus = [data['segmented_title'] + data['segmented_title'] + data['segmented_paragraphs']
                  for data in self.data]
        self.bm25_data = BM25(corpus)
        self.inverted_index_data = InvertedIndex(corpus)

        corpus = []
        for idx, data in enumerate(self.data):
            corpus += [data['segmented_title'] + file for file in data['segmented_file_name']]
        self.bm25_file = BM25(corpus)
        self.inverted_index_file = InvertedIndex(corpus)

        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(os.path.join(MODEL_PATH, 'cws.model'))  # 加载模型

    @staticmethod
    def _get_data():
        with open(os.path.join(DATA_PATH, 'preprocessed.json'), 'r', encoding='utf-8')as f:
            lines = f.readlines()
        return [json.loads(s) for s in lines]

    def search_data(self, query, level):
        query = list(self.segmentor.segment(query))
        result_idx = self.inverted_index_data.search(query)  # 倒排索引查询相关文档
        scores = self.bm25_data.get_scores_in_result(query, result_idx)  # bm25排序
        result = np.argsort(-np.array(scores))
        zero_idx = len(result)
        for i in range(len(result)):
            if scores[result[i]] == 0:
                zero_idx = i
                break
        return [self.data[idx] for idx in result[:zero_idx] if self.data[idx]['level'] <= level]

    def search_file(self, query, level):
        query = list(self.segmentor.segment(query))
        result_idx = self.inverted_index_file.search(query)  # 倒排索引查询相关文档
        scores = self.bm25_file.get_scores_in_result(query, result_idx)  # bm25排序
        result = np.argsort(-np.array(scores))
        zero_idx = len(result)
        for i in range(len(result)):
            if scores[result[i]] == 0:
                zero_idx = i
                break
        result = [self.idx2file[idx] for idx in result[:zero_idx]]
        result = [
            (self.data[data_idx]['file_name'][file_idx], self.data[data_idx]['title'], self.data[data_idx]['level'])
            for data_idx, file_idx in result if self.data[data_idx]['level'] <= level]
        return result


if __name__ == '__main__':
    ret = Retriever()
    a = ret.search_data('大学')
    b = ret.search_file('国家')

    print(a)
    print()
    print(b)

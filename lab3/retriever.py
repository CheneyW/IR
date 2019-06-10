#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/9
"""

import os
import json
import jieba
import numpy as np
from bm25 import BM25

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, 'data')
ATTACHMENT_PATH = os.path.join(DATA_PATH, 'attachment')
MODEL_PATH = os.path.join(DIR_PATH, 'model')

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)
bm25_data_path = os.path.join(MODEL_PATH, 'bm25_data.json')
bm25_file_path = os.path.join(MODEL_PATH, 'bm25_file.json')


class Retriever(object):
    def __init__(self):
        self.data = self._get_data()
        self.idx2file = []
        for idx, data in enumerate(self.data):
            self.idx2file += [(idx, i) for i in range(len(data['segmented_file_name']))]

        self._bm25_build()
        self.bm25_data = BM25(None, path=bm25_data_path)
        self.bm25_file = BM25(None, path=bm25_file_path)

    @staticmethod
    def _get_data():
        with open(os.path.join(DATA_PATH, 'preprocessed.json'), 'r', encoding='utf-8')as f:
            lines = f.readlines()
        return [json.loads(s) for s in lines]

    def _bm25_build(self):
        if not os.path.exists(bm25_data_path):
            corpus = [data['segmented_title'] + data['segmented_paragraphs'] for data in self.data]
            bm25_data = BM25(corpus)
            bm25_data.save(bm25_data_path)
        if not os.path.exists(bm25_file_path):
            corpus = []
            for idx, data in enumerate(self.data):
                corpus += [data['segmented_title'] + file for file in data['segmented_file_name']]
            bm25_file = BM25(corpus)
            bm25_file.save(bm25_file_path)

    def search_data(self, query):
        query = jieba.lcut(query)
        scores = self.bm25_data.get_scores(query)
        result = np.argsort(-np.array(scores))
        zero_idx = len(result)
        for i in range(len(result)):
            if scores[result[i]] == 0:
                zero_idx = i
                break
        return [self.data[idx] for idx in result[:zero_idx]]

    def search_file(self, query):
        query = jieba.lcut(query)
        scores = self.bm25_file.get_scores(query)
        result = np.argsort(-np.array(scores))
        zero_idx = len(result)
        for i in range(len(result)):
            if scores[result[i]] == 0:
                zero_idx = i
                break
        result = [self.idx2file[idx] for idx in result[:zero_idx]]
        result = [(self.data[data_idx]['file_name'][file_idx], self.data[data_idx]['title'])
                  for data_idx, file_idx in result]
        return result


if __name__ == '__main__':
    ret = Retriever()
    a = ret.search_data('大学')
    b = ret.search_file('国家')

    print(a)
    print()
    print(b)

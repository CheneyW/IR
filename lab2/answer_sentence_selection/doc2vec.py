#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/21
"""

import sys
import numpy as np
from pyltp import Segmentor
import os
import json
import gensim
from gensim.models import Doc2Vec

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'data')
MODEL_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'model')

segmentor = Segmentor()  # 初始化实例
segmentor.load(os.path.join(MODEL_PATH, 'ltp', 'cws.model'))  # 加载模型


def train():
    with open(os.path.join(DIR_PATH, 'data.json'), 'r', encoding='utf-8') as f:
        preprocessed_data = [json.loads(x) for x in f.readlines()]

    corpus = []
    for idx, data in enumerate(preprocessed_data):
        sents = data['document'] + [data['question']]
        corpus += [gensim.models.doc2vec.TaggedDocument(list(segmentor.segment(sent)), ['-'.join([str(idx), str(i)])])
                   for i, sent in enumerate(sents)]

    model = Doc2Vec(corpus, dm=1, size=100, window=8, min_count=5, workers=4)
    model.save(os.path.join(DIR_PATH, 'd2v.model'))


if __name__ == '__main__':
    # train()
    model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(DIR_PATH, 'd2v.model'))
    print(model.docvecs.similarity('0-2', '0-1'))
    print(model.docvecs['0-1'])
    print(model.docvecs['0-2'])
    print(type((model.docvecs['0-2'])))
    print(model.docvecs['0-1'] - model.docvecs['0-2'])
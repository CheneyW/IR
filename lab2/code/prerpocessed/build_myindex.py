#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/21
"""
import jieba
from gensim.summarization import bm25
import os
import heapq
import json
from pyltp import Segmentor

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'data')
MODEL_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'model')

idx_path = os.path.join(MODEL_PATH, 'myindex')
if not os.path.exists(idx_path):
    os.mkdir(idx_path)

with open(os.path.join(DATA_PATH, 'stopwords.txt'), 'r', encoding='utf-8') as f:
    stopwords = set(x.strip() for x in f.readlines())

segmentor = Segmentor()  # 初始化实例
segmentor.load(os.path.join(MODEL_PATH, 'ltp', 'cws.model'))  # 加载模型


def seg(sent):
    return [word for word in jieba.cut(sent) if word not in stopwords]  # 0.8897608370702541


def get_data():
    with open(os.path.join(DATA_PATH, 'passages_multi_sentences.json'), 'r', encoding='utf-8') as f:
        pids, docs = [], []
        for line in f.readlines():
            if len(line.strip()) != 0:
                data = json.loads(line)
                pids.append(data['pid'])
                doc = []
                for sent in data['document']:
                    doc += seg(sent)
                docs.append(doc)
    return pids, docs


def get_train_data():
    train_data = []
    with open(os.path.join(DATA_PATH, 'train.json'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))
    return train_data


def build(docs):
    bm25_model = bm25.BM25(docs)
    return bm25_model


def predict(bm25_model):
    results = dict()
    count = 0
    for data in get_train_data():
        if count % 100 == 0:
            print(count)
        count += 1
        query = seg(data['question'])
        scores = bm25_model.get_scores(query)
        rel_idx = list(map(scores.index, heapq.nlargest(3, scores)))
        results[data['qid']] = [pids[i] for i in rel_idx]
    with open(os.path.join(DATA_PATH, 'result2.json'), 'w', encoding='utf-8')as f:
        json.dump(results, f)


def evaluate():
    with open(os.path.join(DATA_PATH, 'result2.json'), 'r', encoding='utf-8')as f:
        results = json.load(f)
    sample_num = len(results)

    correct = [0, 0, 0]
    for data in get_train_data():
        result = results[str(data['qid'])]
        pid = data['pid']
        for i in range(len(result)):
            if pid == result[i]:
                correct[i] += 1
                break

    print('Top1: ', correct[0], '/', sample_num, '=', correct[0] / sample_num)
    print('Top2: ', sum(correct[:2]), '/', sample_num, '=', sum(correct[:2]) / sample_num)
    print('Top3: ', sum(correct), '/', sample_num, '=', sum(correct) / sample_num)
    fail_num = sample_num - sum(correct)
    print('fail: ', fail_num, '/', sample_num, '=', fail_num / sample_num)


if __name__ == '__main__':
    pids, docs = get_data()
    model = build(docs)
    predict(model)
    evaluate()

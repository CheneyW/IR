#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/21

建立基于bm25的检索系统

Top1:  4767 / 5352 = 0.890695067264574
Top2:  4960 / 5352 = 0.9267563527653214
Top3:  5020 / 5352 = 0.9379671150971599

"""
import jieba
import os
import heapq
import json

from prerpocessed.test import BM25

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, 'data')
MODEL_PATH = os.path.join(DIR_PATH, os.pardir, 'model')

result_path = os.path.join(DIR_PATH, 'result_my.json')

bm25_path = os.path.join(DIR_PATH, 'bm25.json')

with open(os.path.join(DATA_PATH, 'stopwords.txt'), 'r', encoding='utf-8') as f:
    stopwords = set(x.strip() for x in f.readlines())


# 分词
def seg(sent):
    return [word for word in jieba.cut(sent) if word not in stopwords]


# 获取文本
def get_docs():
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


# 获取训练数据
def get_train_data():
    train_data = []
    with open(os.path.join(DATA_PATH, 'train.json'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))
    return train_data


def build(docs):
    bm25_model = BM25(docs)
    bm25_model.save(bm25_path)


def predict(pids):
    bm25_model = BM25(None, path=bm25_path)
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
    with open(result_path, 'w', encoding='utf-8')as f:
        json.dump(results, f)


def evaluate():
    with open(result_path, 'r', encoding='utf-8')as f:
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
    pids, docs = get_docs()
    build(docs)
    predict(pids)
    evaluate()

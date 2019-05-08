#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/6
"""

import os
import json
from jieba.analyse import ChineseAnalyzer
from whoosh.index import create_in
from whoosh.fields import *
from whoosh.index import open_dir
from whoosh.qparser import QueryParser
from whoosh import qparser

DATA_PATH = os.path.join(os.pardir, os.pardir, 'data')
MODEL_PATH = os.path.join(os.pardir, os.pardir, 'model')

idx_path = os.path.join(MODEL_PATH, 'index')
if not os.path.exists(idx_path):
    os.mkdir(idx_path)

analyzer = ChineseAnalyzer()


def get_data():
    with open(os.path.join(DATA_PATH, 'passages_multi_sentences.json'), 'r', encoding='utf-8') as f:
        data = []
        for line in f.readlines():
            if len(line.strip()) != 0:
                data.append(json.loads(line))
    return data


def build_index():
    # 定义索引schema
    schema = Schema(pid=ID(stored=True),  # passage id
                    # sid=ID(stored=True),  # sentence id
                    sent=TEXT(stored=True, analyzer=analyzer))  # sentence
    # 创建索引对象
    idx = create_in(idx_path, schema)
    # 添加文档
    writer = idx.writer()
    for data in get_data():
        pid = str(data['pid'])
        if data['pid'] % 1000 == 0:
            print(data['pid'])
        writer.add_document(pid=pid, sent=' '.join(data['document']))

    writer.commit()  # save
    return idx


def get_train_data():
    train_data = []
    with open(os.path.join(DATA_PATH, 'train.json'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            train_data.append(json.loads(line))
    return train_data


def predict():
    idx = open_dir(idx_path)
    results = dict()
    with idx.searcher() as searcher:
        og = qparser.OrGroup.factory(0.9)  # 使包含更多查询项的文档得分更高
        parser = QueryParser("sent", schema=idx.schema, group=og)
        for data in get_train_data():
            query = parser.parse(data['question'])
            results[data['qid']] = [int(dict(hit)['pid']) for hit in searcher.search(query)[:3]]

    with open(os.path.join('data', 'result1.json'), 'w', encoding='utf-8')as f:
        json.dump(results, f)


def evaluate():
    with open(os.path.join(DATA_PATH, 'result1.json'), 'r', encoding='utf-8')as f:
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

    print('question num: ', sample_num)
    print('first correct: ', correct[0], '/', sample_num, '=', correct[0] / sample_num)
    print('second correct: ', correct[1], '/', sample_num, '=', correct[1] / sample_num)
    print('third correct: ', correct[2], '/', sample_num, '=', correct[2] / sample_num)
    fail_num = sample_num - sum(correct)
    print('fail: ', fail_num, '/', sample_num, '=', fail_num / sample_num)


if __name__ == '__main__':
    # predict()
    evaluate()

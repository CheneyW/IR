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

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, 'data')

result_path = os.path.join(DIR_PATH, 'result_whoosh.json')

idx_path = os.path.join(DIR_PATH, 'index')
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
    # build_index()
    predict()
    evaluate()

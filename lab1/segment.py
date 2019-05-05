#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/1
"""
from pyltp import Segmentor
import json
import os

CWS_MODEL_PATH = os.path.join('ltp', 'cws.model')  # 模型路径
STOPWORD_PATH = os.path.join('data', 'stopwords.txt')  # 停用词
DATA_PATH = os.path.join('data', 'data.json')  # 数据路径
OUTPUT_PATH = os.path.join('data', 'preprocessed.json')  # 输出路径


def get_stopword():
    with open(STOPWORD_PATH, 'r', encoding='utf-8') as f:
        sw = set(x.strip() for x in f.readlines())
    return sw


def get_data():
    data = []
    with open(DATA_PATH, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line.strip()) != 0:
                data.append(json.loads(line))
    return data


def preprocess():
    stopword = get_stopword()
    data_lst = get_data()
    segmentor = Segmentor()  # 初始化实例
    segmentor.load(CWS_MODEL_PATH)  # 加载模型

    for data in data_lst:
        data['segmented_title'] = [x for x in segmentor.segment(data['title']) if x not in stopword]
        data['segmented_paragraphs'] = [x for x in segmentor.segment(data['paragraphs']) if x not in stopword]
        del data['title']
        del data['paragraphs']
    return data_lst


if __name__ == '__main__':
    with open(os.path.join(OUTPUT_PATH), 'w', encoding='utf-8') as f:
        for d in preprocess():
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')

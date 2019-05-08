#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/7
"""

import os
import json
from pyltp import Segmentor
from pyltp import Postagger

DATA_PATH = os.path.join(os.pardir, os.pardir, 'data')
MODEL_PATH = os.path.join(os.pardir, os.pardir, 'model')


class LTP(object):
    def __init__(self):
        # 停用词
        with open(os.path.join(DATA_PATH, 'stopwords.txt'), 'r', encoding='utf-8') as f:
            self.stopwords = set(x.strip() for x in f.readlines())
        # 分词
        self.segmentor = Segmentor()  # 初始化实例
        segmentor.load(os.path.join(MODEL_PATH, 'ltp', 'cws.model'))  # 加载模型
        # 词性标注
        self.postagger = Postagger()  # 初始化实例
        postagger.load(os.path.join(MODEL_PATH, 'ltp', 'pos.model'))  # 加载模型

    def seg(self, sent):
        return list(self.segmentor.segment(sent))

    def pos(self, words):
        return list(self.postagger.postag(words))

    def stop_words(self, words):
        return [x for x in words if x not in self.stop_words()]


def preprocess():
    def get_train_data():
        train_data = []
        count = 0
        with open(os.path.join(DATA_PATH, 'train.json'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                d = json.loads(line)
                if len(d['answer_sentence']) > 1:
                    count += 1
                train_data.append(d)
                # train_data.append(json.loads(line))
        print(count)
        return train_data

    def get_passages():
        passages = []
        with open(os.path.join(DATA_PATH, 'passages_multi_sentences.json'), 'r', encoding='utf-8') as f:
            for line in f.readlines():
                if len(line.strip()) != 0:
                    passages.append(json.loads(line)['document'])
        return passages

    train_data = get_train_data()
    passages = get_passages()
    preprocessed_data = []
    for train in train_data:
        data = dict()
        # 文档
        data['document'] = passages[train['pid']]
        # 答案句的索引
        answer_idx = []
        try:
            for sent in train['answer_sentence']:
                idx = data['document'].index(sent)
                if idx not in answer_idx:
                    answer_idx.append(idx)
        except:
            continue
        data['answer_idx'] = answer_idx
        # 问题
        data['question'] = train['question']
        preprocessed_data.append(data)

    with open('data/result2.json', 'w', encoding='utf-8') as f:
        for data in preprocessed_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


# 提取特征
def extract_feature():
    ltp = LTP()

    # 提取答句自身特征
    def feature_self(s, seg):
        new_feature = []
        new_feature.append(1) if ':' in s or '：' in s else new_feature.append(0)  # 是否包含冒号
        new_feature.append(len(seg))  # 句子中词数

    with open('data/result2.json', 'r', encoding='utf-8') as f:
        preprocessed_data = f.readlines()

    svm_rank_train = []
    for i in range(len(preprocessed_data)):
        data = json.loads(preprocessed_data[i])
        answer_idx = data['answer_idx']
        question = data['question']
        for j in range(len(data['document'])):
            sent = data['document'][j]
            features = []

            s = '1' if j in answer_idx else '0'
            s += ' qid:' + str(i)
            for k in range(len(features)):
                s += ' ' + str(k + 1) + ':' + str(features[k])
            svm_rank_train.append(s)

    with open('data/svmrank_train.dat', 'w', encoding='utf-8')as f:
        f.writelines(svm_rank_train)


if __name__ == '__main__':
    preprocess()

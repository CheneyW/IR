#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/10
"""

import os
import json

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'data')


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
    return train_data


def get_passages():
    passages = []
    with open(os.path.join(DATA_PATH, 'passages_multi_sentences.json'), 'r', encoding='utf-8') as f:
        for line in f.readlines():
            if len(line.strip()) != 0:
                passages.append(json.loads(line)['document'])
    return passages


def preprocess():
    train_data = get_train_data()
    passages = get_passages()
    preprocessed_data = []
    error_count = 0
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
            error_count+=1
            continue
        data['answer_idx'] = answer_idx
        # 问题
        data['question'] = train['question']
        preprocessed_data.append(data)
    print(error_count)
    # svm rank
    with open(os.path.join(DIR_PATH, 'data.json'), 'w', encoding='utf-8') as f:
        for data in preprocessed_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')

    # # bert
    # bert_train_data = []
    # # 70% 训练集 10% 验证集 20%测试集
    # train_range = int(len(preprocessed_data) * 0.7)
    # dev_range = int(len(preprocessed_data) * 0.8)
    #
    # for data in preprocessed_data:
    #     answer_idx = data['answer_idx']
    #     doc = []
    #     for idx, sent in enumerate(data['document']):
    #         s = '1' if idx in answer_idx else '0'
    #         s += '\t' + data['question'] + '\t' + sent + '\n'
    #         doc.append(s)
    #     bert_train_data.append(doc)
    #
    # with open(os.path.join(DIR_PATH, 'sent.dat'), 'w', encoding='utf-8') as f:
    #     for doc in bert_train_data:
    #         f.writelines(doc)
    #
    # with open(os.path.join(DIR_PATH, 'sent.train.dat'), 'w', encoding='utf-8') as f:
    #     for doc in bert_train_data[:train_range]:
    #         f.writelines(doc)
    #
    # with open(os.path.join(DIR_PATH, 'sent.dev.dat'), 'w', encoding='utf-8') as f:
    #     for doc in bert_train_data[train_range:dev_range]:
    #         f.writelines(doc)
    #
    # with open(os.path.join(DIR_PATH, 'sent.test.dat'), 'w', encoding='utf-8') as f:
    #     for doc in bert_train_data[dev_range:]:
    #         f.writelines(doc)


if __name__ == '__main__':
    preprocess()

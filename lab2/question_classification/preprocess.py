#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/11

将问题分类的标注转化为数字表示
"""

import os
from sklearn import preprocessing

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(DIR_PATH, 'trian_questions.txt')
test_path = os.path.join(DIR_PATH, 'test_questions.txt')


def get_data(path):
    labels, subtype_labels, texts = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            content = line.strip().split('\t')
            texts.append(content[1])
            subtype_labels.append(content[0])
            labels.append(content[0][:3])
    return texts, labels, subtype_labels


if __name__ == '__main__':
    """
    将问题分类的标注转化为数字表示
    文件的第一列为类别；第二列为子类别；第三列为文本
    """
    texts, labels, subtype_labels = get_data(train_path)
    label_encoder = preprocessing.LabelEncoder()
    subtype_label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    subtype_labels = subtype_label_encoder.fit_transform(subtype_labels)
    with open(os.path.join(DIR_PATH, 'train_que.dat'), 'w', encoding='utf-8')as f:
        for i in range(len(texts)):
            f.write(str(labels[i]) + '\t' + str(subtype_labels[i]) + '\t' + texts[i] + '\n')

    texts, labels, subtype_labels = get_data(test_path)
    labels = label_encoder.transform(labels)
    subtype_labels = subtype_label_encoder.transform(subtype_labels)
    with open(os.path.join(DIR_PATH, 'test_que.dat'), 'w', encoding='utf-8')as f:
        for i in range(len(texts)):
            f.write(str(labels[i]) + '\t' + str(subtype_labels[i]) + '\t' + texts[i] + '\n')

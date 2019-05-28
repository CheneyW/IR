#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/10
"""
import os

import jieba
from pyltp import Segmentor, Postagger, NamedEntityRecognizer

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'data')
MODEL_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'model')


class LTP(object):
    def __init__(self):
        # 停用词
        with open(os.path.join(DATA_PATH, 'stopwords.txt'), 'r', encoding='utf-8') as f:
            self.stopwords = set(x.strip() for x in f.readlines())
        # 分词
        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(os.path.join(MODEL_PATH, 'ltp', 'cws.model'))  # 加载模型
        # 词性标注
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(os.path.join(MODEL_PATH, 'ltp', 'pos.model'))  # 加载模型
        # 命名实体识别
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(MODEL_PATH, 'ltp', 'ner.model'))

    def seg(self, sent):
        # return jieba.lcut(sent)
        return list(self.segmentor.segment(sent))

    def pos(self, words):
        return list(self.postagger.postag(words))

    def recognize(self, words, postags):
        return list(self.recognizer.recognize(words, postags))

    def stop_words(self, words):
        return [x for x in words if x not in self.stop_words()]


if __name__ == '__main__':
    ltp = LTP()

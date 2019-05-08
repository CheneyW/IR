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

LTP_MODEL = os.path.join('model', 'ltp')


class LTP(object):
    def __init__(self):
        # 停用词
        with open(os.path.join('data', 'stopwords.txt'), 'r', encoding='utf-8') as f:
            self.stopwords = set(x.strip() for x in f.readlines())
        # 分词
        self.segmentor = Segmentor()  # 初始化实例
        self.segmentor.load(os.path.join(LTP_MODEL, 'cws.model'))  # 加载模型
        # 词性标注
        self.postagger = Postagger()  # 初始化实例
        self.postagger.load(os.path.join(LTP_MODEL, 'pos.model'))  # 加载模型

    def seg(self, sent):
        return list(self.segmentor.segment(sent))

    def pos(self, words):
        return list(self.postagger.postag(words))

    def stop_words(self, words):
        return [x for x in words if x not in self.stopwords]


if __name__ == '__main__':
    ltp = LTP()
    s = '明日的今日子是什么时候开始连载的'
    words = ltp.seg(s)
    pos = ltp.pos(words)
    sw = ltp.stop_words(words)

    print(words)
    print(pos)
    print(sw)

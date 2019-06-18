#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/18

倒排索引
"""


class InvertedIndex(object):
    def __init__(self, corpus):
        self.dic = dict()
        for idx, sent in enumerate(corpus):
            for word in sent:
                if word not in self.dic:
                    self.dic[word] = set()
                self.dic[word].add(idx)

    def search(self, query):
        result = set()
        for word in query:
            if word in self.dic:
                result = result | self.dic[word]
        return result

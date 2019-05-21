#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/19
"""


class BOW(object):
    def __init__(self):
        self.dic = dict()

    def fit(self, docs):
        for doc in docs:
            for w in doc:
                self.add(w)

    def add(self, word):
        if word not in self.dic.keys():
            self.dic[word] = len(self.dic)

    def transform(self, docs):
        out = [[0] * len(self.dic) for _ in range(len(docs))]
        for i, doc in enumerate(docs):
            for j, w in enumerate(doc):
                if w in self.dic.keys():
                    out[i][self.dic[w]] = 1
        return out

    def vocab(self):
        return self.dic


if __name__ == '__main__':
    train = [['我', '爱', '北京', '天安门'], ['矿泉水', '和', '果汁', '哪个', '容易', '结冰']]
    test = [['我', '在', '天安门', '卖', '矿泉水']]
    bow = BOW()
    bow.fit(train)
    print(bow.vocab())
    print(bow.transform(train))
    print(bow.transform(test))
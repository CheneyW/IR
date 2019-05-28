#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/20
"""
import numpy as np


class Item(object):
    def __init__(self, qid):
        self.qid = qid
        self.sent_num = 0
        self.probs = []
        self.ans_idx = []

    def add(self, label, prob):
        if label == 1:
            self.ans_idx.append(self.sent_num)
        self.probs.append(prob)
        self.sent_num += 1

    def result(self):  # 根据概率排序并计算预测结果
        arg = np.array(self.probs).argsort().tolist()
        rank = [len(arg) - arg.index(idx) for idx in self.ans_idx]
        # mrr = sum([1 / r for r in rank]) / len(self.ans_idx)
        top_rank = min(rank)
        top_n = [1 if top_rank <= i + 1 else 0 for i in range(5)]
        mrr = 1 / top_rank
        rank.sort()
        map = sum([i + 1 / r for i, r in enumerate(rank)]) / len(self.ans_idx)
        return mrr, map, top_n


def evaluate():
    with open('test.dat', 'r', encoding='utf-8')as f:
        sample = f.readlines()
    with open('predictions.dat', 'r', encoding='utf-8')as f:
        predict = f.readlines()

    ques = dict()
    for i, line in enumerate(sample):
        content = line.split()
        if len(content) < 2:
            continue
        label = int(content[0])
        qid = int(content[1][4:])
        prob = float(predict[i].strip())

        if qid not in ques.keys():
            ques[qid] = Item(qid)
        ques[qid].add(label, prob)

    mrr, map, top = 0, 0, [0] * 5
    for x in ques.values():
        a, b, top_n = x.result()
        mrr += a
        map += b
        top = [top[i] + top_n[i] for i in range(5)]
    mrr /= len(ques)
    map /= len(ques)
    top_n = [x / len(ques) for x in top]

    return mrr, map, top_n


if __name__ == '__main__':
    mrr, map, top = evaluate()
    print(mrr, map)
    print('\n'.join(['top%d : %f' % (idx + 1, x) for idx, x in enumerate(top)]))



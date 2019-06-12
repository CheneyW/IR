#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
SVM_PATH = os.path.join(DIR_PATH, 'svm_rank')

g_count = 0
split_idx = 4500


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

    def get_rank(self):
        rank_idx = list(reversed(np.array(self.probs).argsort().tolist()))
        print(rank_idx[0], self.ans_idx)
        if rank_idx[0] in self.ans_idx:
            global g_count
            g_count += 1
        return rank_idx


def evaluate():
    with open(os.path.join(SVM_PATH, 'test.dat'), 'r', encoding='utf-8')as f:
        sample = f.readlines()
    with open(os.path.join(SVM_PATH, 'predictions.dat'), 'r', encoding='utf-8')as f:
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
    results = []
    for x in ques.values():
        results.append(x.get_rank()[:3])
        a, b, top_n = x.result()
        mrr += a
        map += b
        top = [top[i] + top_n[i] for i in range(5)]
    mrr /= len(ques)
    map /= len(ques)
    top_n = [x / len(ques) for x in top]

    data = []
    with open(os.path.join(DIR_PATH, 'data.json'), 'r', encoding='utf-8')as f:
        for i, line in enumerate(f.readlines()[split_idx:]):
            d = json.loads(line)
            d['answer_sentence'] = [d['document'][x] for x in results[i]]
            del d['document']
            data.append(d)

    with open(os.path.join(DIR_PATH, 'result.json'), 'w', encoding='utf-8')as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')

    print(g_count, len(ques))
    return mrr, map, top_n


def run():
    """
    最终的文件测试
    """
    with open(os.path.join(SVM_PATH, 'test.dat'), 'r', encoding='utf-8')as f:
        sample = f.readlines()
    with open(os.path.join(SVM_PATH, 'predictions.dat'), 'r', encoding='utf-8')as f:
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

    results = []
    for x in ques.values():
        results.append(x.get_rank()[0])

    with open(os.path.join(DIR_PATH, 'result_task1.json'), 'r', encoding='utf-8')as f:
        data = [json.loads(line) for line in f.readlines()]
    for i, d in enumerate(data):
        d['answer_sentence'] = [d['document'][results[i]]]
        del d['document']

    with open(os.path.join(DIR_PATH, 'result_task3.json'), 'w', encoding='utf-8')as f:
        for d in data:
            json.dump(d, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    mrr, map, top = evaluate()
    print('MRR:', mrr)
    print('MAP:', map)
    print('\n'.join(['top%d : %f' % (idx + 1, x) for idx, x in enumerate(top)]))

    # run()

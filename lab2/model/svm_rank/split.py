#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/10
"""
import os
import numpy as np

from model.svm_rank.evaluate import evaluate


# split_idx = int(5316/5*4)
split_idx = 4500

def split():
    train = []
    test = []
    with open('svmrank_train.dat', 'r', encoding='utf-8') as f:
        for line in f.readlines():
            qid = line.split()[1][4:]
            if int(qid) < split_idx:
                train.append(line)
            else:
                test.append(line)

    with open('train.dat', 'w', encoding='utf-8')as f:
        f.writelines(train)

    with open('test.dat', 'w', encoding='utf-8')as f:
        f.writelines(test)


if __name__ == '__main__':
    split()
    # idx_lst, val_lst = [], []
    # for i in range(1, 15):
    #     t = 20 * i
    #     # 训练
    #     s = 'svm_rank_learn.exe -c %d train.dat model' % t
    #     print(s)
    #     os.system(s)
    #     # 测试
    #     os.system('svm_rank_classify.exe test.dat model predictions.dat')
    #     val = evaluate()[0]
    #     idx_lst.append(t)
    #     val_lst.append(val)
    # print(idx_lst)
    # print(val_lst)
    # print(zip(idx_lst, val_lst))

    os.system('svm_rank_learn.exe -c 200 train.dat model')
    # os.system('svm_rank_learn.exe -c 5 train.dat model')
    os.system('svm_rank_classify.exe test.dat model predictions.dat')


# 0.73135527607123 0.7382526128555182
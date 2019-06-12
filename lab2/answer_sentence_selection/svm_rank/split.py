#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/10
"""
import os

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
    # split()

    os.system('svm_rank_learn.exe -c 200 train.dat model')

    os.system('svm_rank_classify.exe test.dat model predictions.dat')

# 0.7333559480743967 0.7406188458548961

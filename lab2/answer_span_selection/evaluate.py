#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/3
"""
import json

from answer_span_selection.metric import bleu1, exact_match


def evaluate(path):
    with open(path, 'r', encoding='utf-8')as f:
        output = [json.loads(line) for line in f.readlines()]
    bleu = [[] for _ in range(6)]
    predictions, truth = [], []
    for data in output:
        predictions.append(data['prediction'])
        truth.append(data['answer'])
        bleu[int(data['que_type'])].append(bleu1(data['prediction'], data['answer']))
    bleu_val = sum([sum(x) for x in bleu]) / sum([len(x) for x in bleu])
    print(bleu_val)
    print(exact_match(predictions, truth))

    # for i, x in enumerate(bleu):
    #     print('type%d : %f' % (i, (sum(x) / len(x)) if len(x) != 0 else 0))

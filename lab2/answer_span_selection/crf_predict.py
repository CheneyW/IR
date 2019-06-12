#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/3
"""

import os
import json
from answer_span_selection.evaluate import evaluate
from answer_span_selection.rule import RuleBased
from question_classification.ltp import LTP

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
CRF_PATH = os.path.join(DIR_PATH, 'CRF++-0.58')


def train():
    os.system('CRF++-0.58\crf_learn -c 5.0 CRF++-0.58\\template CRF++-0.58\crf_train.dat CRF++-0.58\model')


def predict(output_path, rule=True):
    """
    :param output_path: 答案抽取输出路径
    :param rule: True:crf和规则融合 ； False:仅crf
    """
    os.system('CRF++-0.58\crf_test -m CRF++-0.58\model CRF++-0.58\crf_test.dat > CRF++-0.58\output.dat')
    with open(os.path.join(CRF_PATH, 'output.dat'), 'r', encoding='utf-8')as f:
        output = f.readlines()
    with open(os.path.join(DIR_PATH, 'test.json'), 'r', encoding='utf-8')as f:
        test_data = [json.loads(line) for line in f.readlines()]

    samples, sample_idx = [''] * len(test_data), 0
    type, types = '', [0] * len(test_data)
    for line in output:
        content = line.strip()
        if len(content) == 0:
            types[sample_idx] = int(type)
            sample_idx += 1
            continue
        content = content.split()
        word, tag = content[0], content[-1]
        type = content[2]
        # if tag != 'O' and word not in samples[sample_idx]:
        if tag != 'O':
            samples[sample_idx] += word

    results = []
    ltp = LTP()
    rb = RuleBased() if rule else None
    for idx, data in enumerate(test_data):
        if len(samples[idx]) == 0:
            if rule:
                samples[idx] = rb.get(data['question'], ''.join(data['answer_sentence']))[0]
            else:
                samples[idx] = ''.join(data['answer_sentence'])

        result = dict()
        result['prediction'] = samples[idx]
        result['answer'] = data['answer']
        # result['qid'] = data['qid']
        result['question'] = data['question']
        result['que_type'] = types[idx]
        results.append(result)

    with open(output_path, 'w', encoding='utf-8')as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')


def run():
    """
    最终的文件测试
    """
    os.system('CRF++-0.58\crf_learn -c 5.0 CRF++-0.58\\template CRF++-0.58\\train.dat CRF++-0.58\model')

    os.system('CRF++-0.58\crf_test -m CRF++-0.58\model CRF++-0.58\\test.dat > CRF++-0.58\output.dat')

    with open(os.path.join(CRF_PATH, 'output.dat'), 'r', encoding='utf-8')as f:
        output = f.readlines()
    with open(os.path.join(DIR_PATH, 'result_task3.json'), 'r', encoding='utf-8')as f:
        test_data = [json.loads(line) for line in f.readlines()]

    samples, sample_idx = [''] * len(test_data), 0
    type, types = '', [0] * len(test_data)
    for line in output:
        content = line.strip()
        if len(content) == 0:
            types[sample_idx] = int(type)
            sample_idx += 1
            continue
        content = content.split()
        word, tag = content[0], content[-1]
        type = content[2]
        # if tag != 'O' and word not in samples[sample_idx]:
        if tag != 'O':
            samples[sample_idx] += word

    rb = RuleBased()
    for idx, data in enumerate(test_data):
        if len(samples[idx]) == 0:
            samples[idx] = rb.get(data['question'], ''.join(data['answer_sentence']))[0]

        data['answer'] = samples[idx]
        del data['answer_sentence']

    with open(os.path.join(DIR_PATH, 'result_task4.json'), 'w', encoding='utf-8')as f:
        for data in test_data:
            json.dump(data, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    # train()
    output_path = os.path.join(DIR_PATH, 'output_crf.json')
    predict(output_path, rule=True)
    evaluate(output_path)

    # run()

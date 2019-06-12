#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/3
"""
import os
import json

from question_classification.ltp import LTP
from question_classification.que_clf import QuestionClassifier

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, 'data')
MODEL_PATH = os.path.join(DIR_PATH, os.pardir, 'model')

ltp = LTP()


def crf_train(data):
    """
    提取crf特征
    """
    qc = QuestionClassifier()
    output = []
    for q in data:
        sample = []
        # 问题
        que_seg = ltp.seg(q['question'])
        que_type, que_subtype = qc.predict_sent(q['question'])
        # 答案句
        sent_seg, sent_pos, sent_ner, sent_arcs = ltp.get_all(' '.join(q['answer_sentence']))
        # 答案
        ans_seg = ltp.seg(q['answer'])
        ans_idx = get_label(sent_seg, ans_seg)
        # if len(ans_idx) == 0:
        #     print(q['answer'], q['answer_sentence'])
        #     continue

        after_colon_idx, i = [], 0
        while i < len(sent_seg):
            if not (sent_seg[i] == ':' or sent_seg[i] == '：'):
                i += 1
                continue
            i += 1
            while i < len(sent_seg) and sent_seg[i] not in ['，', '。', '！', '？']:
                after_colon_idx.append(i)
                i += 1
        que_idx_in_sent = [i for i, word in enumerate(sent_seg) if word in que_seg]
        for idx, word in enumerate(sent_seg):
            if idx in ans_idx:
                label = 'I-ANS' if idx > 0 and idx - 1 in ans_idx else 'B-ANS'
                # label = 'ANS'
            else:
                label = 'O'

            line = [word]
            line.append('1' if word in que_seg else '0')
            line.append(que_type)  # 在模板中使用
            line.append(que_subtype)  # 在模板中使用
            line.append(sent_pos[idx])
            line.append(sent_ner[idx])
            line.append(sent_arcs[idx].relation)
            line.append('/'.join([que_type, sent_pos[idx]]))
            line.append('/'.join([que_type, sent_ner[idx]]))
            line.append('/'.join([que_type, sent_arcs[idx].relation]))
            line.append('1' if word == ':' or word == '：' else '0')
            line.append('1' if idx in after_colon_idx else '0')
            line.append(str((idx + 1) / len(sent_seg)))
            temp = [abs(x - idx) for x in que_idx_in_sent]
            line.append(str(min(temp)) if len(temp) != 0 else '-1')
            line.append(str(sum(temp) / len(temp)) if len(temp) != 0 else '-1')
            line.append(label)
            sample.append('\t'.join(line))

        output.append('\n'.join(sample) + '\n\n')
    return output


def get_label(sent_seg, ans_seg):
    """
    :param sent_seg: 分词后的答案句
    :param ans_seg: 分词后的答案
    :return: 答案句中作为答案的索引
    """
    sent, ans = ''.join(sent_seg), ''.join(ans_seg)
    ans_idx = []
    if ans in sent:
        begin = sent.index(ans)
        end = begin + len(ans) - 1

        pos = [[0, 0] for _ in range(len(sent_seg))]
        for idx, word in enumerate(sent_seg):
            pos[idx][1] = pos[idx][0] + len(word) - 1
            if idx + 1 < len(pos):
                pos[idx + 1][0] = pos[idx][0] + len(word)

        for idx, p in enumerate(pos):
            if begin <= p[1] and p[0] <= end:
                ans_idx.append(idx)
    else:
        for idx, word in enumerate(sent_seg):
            if word in ans_seg:
                ans_idx.append(idx)
    return ans_idx


def test():
    """
    训练集答案句标注序列中，统计所有词的标注均不是答案的句子数。
    """
    with open(os.path.join(DATA_PATH, 'train.json'), 'r', encoding='utf-8')as f:
        data = [json.loads(line) for line in f.readlines()]

    count = 0
    for q in data:
        sent_seg = ltp.seg(' '.join(q['answer_sentence']))
        ans_seg = ltp.seg(q['answer'])
        # ans = False
        # for idx, word in enumerate(sent_seg):
        #     if word in ans_seg:
        #         ans = True
        if len(get_label(sent_seg, ans_seg)) == 0:
            count += 1
            print(q['qid'], q['answer_sentence'], q['answer'])

    print(count)
    print(len(data))


if __name__ == '__main__':
    # test()

    with open(os.path.join(DIR_PATH, 'train.json'), 'r', encoding='utf-8')as f:
        train_data = [json.loads(line) for line in f.readlines()]
    with open(os.path.join(DIR_PATH, 'test.json'), 'r', encoding='utf-8')as f:
        test_data = [json.loads(line) for line in f.readlines()]

    train_data = crf_train(train_data)

    test_data = crf_train(test_data)

    with open(os.path.join(DIR_PATH, 'CRF++-0.58', 'crf_train.dat'), 'w', encoding='utf-8')as f:
        f.write('\n'.join(train_data))
    with open(os.path.join(DIR_PATH, 'CRF++-0.58', 'crf_test.dat'), 'w', encoding='utf-8')as f:
        f.write('\n'.join(test_data))

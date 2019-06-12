#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/6/3
"""

import os
import json
import re

from question_classification.ltp import LTP
from question_classification.que_clf import QuestionClassifier
from answer_span_selection.evaluate import evaluate

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, 'data')
MODEL_PATH = os.path.join(DIR_PATH, os.pardir, 'model')


class RuleBased(object):
    def __init__(self):
        self.ltp = LTP()
        self.qc = QuestionClassifier()
        self.qc.load()

    def get(self, question, ans_sent):
        que_type = int(self.qc.predict_sent(question)[0])

        seg = self.ltp.seg(ans_sent)
        pos = self.ltp.pos(seg)

        sent = ''.join(seg)
        result = ''
        if '：' in sent or ':' in sent:
            count = sum([1 if x == '：' or x == ':' else 0 for x in sent])
            if count == 1:
                begin = sent.index('：') if '：' in sent else sent.index(':')  # 冒号
                for idx in range(begin + 1, len(sent)):
                    if sent[idx] in ['，', '。', '！', '？']:
                        break
                    result += sent[idx]
                return result, que_type
            else:
                for idx, word in enumerate(seg):
                    if not (word == ':' or word == '：'):
                        continue
                    if idx > 0 and idx + 1 < len(seg) and pos[idx - 1] == 'm' and pos[idx + 1] == 'm':  # 时间
                        break
                    elif seg[idx - 1] in ans_sent:  # 多个冒号
                        result_seg = []
                        for i in range(idx + 1, len(seg)):
                            if seg[i] in ['，', '。', '！', '？']:
                                break
                            elif seg[i] == ':' or seg[i] == '：':
                                result_seg.pop()
                                break
                            else:
                                result_seg.append(seg[i])
                        return ''.join(result_seg), que_type

        if que_type == 0:
            result = self.rule0(seg, pos)
        elif que_type == 1:
            result = self.rule1(seg, pos)
        elif que_type == 2:
            result = self.rule2(seg, pos)
        elif que_type == 3:
            result = self.rule3(seg, pos)
        elif que_type == 4:
            result = self.rule4(seg, pos)
        elif que_type == 5:
            result = self.rule5(seg, pos)

        if len(result) == 0:
            result = ''.join([w for w in seg if w not in question])
            # result = ans_sent
        return result, que_type

    # 类别0 描述（DES）
    def rule0(self, seg, pos):
        return ''

    # 类别1 人物（HUM）
    def rule1(self, seg, pos):  # 人物
        ner = self.ltp.recognize(seg, pos)
        result = ''
        for idx, ne_tag in enumerate(ner):
            if ne_tag.endswith('Nh') or ne_tag.endswith('Ni'):  # 人名、机构名
                result += seg[idx]
        return result

    # 类别2 地点 （LOC）
    def rule2(self, seg, pos):  # 地点
        ner = self.ltp.recognize(seg, pos)
        result = ''
        for idx, ne_tag in enumerate(ner):
            if ne_tag.endswith('Ns'):  # 地点
                result += seg[idx]
        return result

    # 类别3 数字 (NUM)
    def rule3(self, seg, pos):
        result = ''
        i = 0
        while i < len(pos):
            if pos[i] == 'm':
                result += seg[i]
                i += 1
                while i < len(pos) and (pos[i] == 'm' or pos[i] == 'c'):
                    result += seg[i]
                    i += 1
            else:
                i += 1
        return result

    # 类别4 实体 (OBJ)
    def rule4(self, seg, pos):
        result = ''
        sent = ''.join(seg)
        if '《' in sent:
            result = ''.join(re.findall('《.*?》', sent))
        # for idx, pos_tag in enumerate(pos):
        #     if pos_tag.startswith('n'):
        #         result += seg[idx]
        return result

    # 类别5 时间 (TIME)
    def rule5(self, seg, pos):
        result = ''
        for idx, pos_tag in enumerate(pos):
            if pos_tag == 'nt':
                result += seg[idx]
        return result


def predict(output_path):
    rb = RuleBased()
    with open(os.path.join(DIR_PATH, 'test.json'), 'r', encoding='utf-8')as f:
        test_data = [json.loads(line) for line in f.readlines()]

    results = []
    for data in test_data:
        result = dict()
        # que_type = 0
        # result['prediction'] = ''.join(rb.deal_with_rules(data['question'], ''.join(data['answer_sentence'])))
        result['prediction'], que_type = rb.get(data['question'], ' '.join(data['answer_sentence']))

        result['answer'] = data['answer']
        result['qid'] = data['qid']
        result['question'] = data['question']
        result['que_type'] = que_type
        results.append(result)
    with open(output_path, 'w', encoding='utf-8')as f:
        for result in results:
            json.dump(result, f, ensure_ascii=False)
            f.write('\n')


if __name__ == '__main__':
    output_path = os.path.join(DIR_PATH, 'output_rule.json')
    predict(output_path)
    evaluate(output_path)

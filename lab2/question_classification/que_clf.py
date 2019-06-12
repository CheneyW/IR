#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/10
"""
import os
import joblib
# import synonyms
from sklearn import linear_model, metrics

from question_classification.ltp import LTP
from question_classification.bow import BOW

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, 'data')
MODEL_PATH = os.path.join(DIR_PATH, 'model')

if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

train_path = os.path.join(DIR_PATH, 'train_que.dat')
test_path = os.path.join(DIR_PATH, 'test_que.dat')

que_words = ['谁', '谁是', '何', '什么', '是什么', '哪儿', '哪里', '哪', '哪个', '哪些', '几时', '几', '多少',
             '怎', '怎么', '怎的', '怎样', '怎么样', '怎么着', '如何', '为什么',
             '吗', '呢', '吧', '啊',
             '难道', '岂', '居然', '竟然', '究竟', '简直', '难怪', '反倒', '何尝', '何必']

with open(os.path.join(DATA_PATH, 'stopwords.txt'), 'r', encoding='utf-8') as f:
    stopwords = set(x.strip() for x in f.readlines())


# 特征提取
class QuestionClassifier(object):
    def __init__(self, load=True):
        self.ltp = LTP()
        self.model, self.model_sub = linear_model.LogisticRegression(), linear_model.LogisticRegression()

        # self.model, self.model_sub = svm.SVC(), svm.SVC()
        # self.model, self.model_sub = naive_bayes.MultinomialNB(), naive_bayes.MultinomialNB()
        self.bow = BOW()
        self.count = 0

        if load:
            self.load()

    def train(self, fit_test=False):
        texts, labels, subtype_label = get_data(train_path)
        if fit_test:
            texts2, labels2, subtype_label2 = get_data(test_path)
            texts += texts2
            labels += labels2
            subtype_label += subtype_label2

        train_x = [self.extract(text) for text in texts]
        with open('test.txt', 'w', encoding='utf-8') as f:
            for i in range(len(labels)):
                f.write(str(labels[i]) + '\t' + str(subtype_label[i]) + '\t' + str(train_x[i]) + '\n')
        self.bow.fit(train_x)
        train_x = self.bow.transform(train_x)
        print(len(self.bow.vocab()))
        self.model.fit(train_x, labels)
        self.model_sub.fit(train_x, subtype_label)

        self.bow.save(os.path.join(MODEL_PATH, 'bow.json'))
        joblib.dump(self.model, os.path.join(MODEL_PATH, 'model.pkl'))
        joblib.dump(self.model_sub, os.path.join(MODEL_PATH, 'model_sub.pkl'))

    def load(self):
        self.bow.load(os.path.join(MODEL_PATH, 'bow.json'))
        self.model = joblib.load(os.path.join(MODEL_PATH, 'model.pkl'))
        self.model_sub = joblib.load(os.path.join(MODEL_PATH, 'model_sub.pkl'))

    def predict(self):
        texts, labels, subtype_label = get_data(test_path)
        test_x = [self.extract(text) for text in texts]
        test_x = self.bow.transform(test_x)

        acc1 = metrics.accuracy_score(self.model.predict(test_x), labels)
        acc2 = metrics.accuracy_score(self.model_sub.predict(test_x), subtype_label)
        return acc1, acc2

    def predict_sent(self, sent):
        test_x = self.extract(sent)
        test_x = self.bow.transform([test_x])
        p1 = self.model.predict(test_x)[0]
        p2 = self.model_sub.predict(test_x)[0]
        return p1, p2

    @staticmethod
    def ccw(words, postags, arcs):  # 问题类别线索词集合（ Category Clue Words set ， CCWs ）
        HEDs, SBJs, QWs, RELs = set(), set(), set(), set()  # 问题类别线索词集合（ Category Clue Words set ， CCWs ）
        # 若其依存关系为 HED ，则加入 HEDs 集合
        HEDs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'HED')
        # 若其依存关系为COO,且其父节点是HED节点,则将其加入 HEDs 集合
        HEDs.update(idx for idx, arc in enumerate(arcs) if arc.relation == 'COO' and arc.head - 1 in HEDs)
        # 若其依存关系是SBV,且其父节点是HED节点,且与其父节点相邻，则将其加入 SBJs 集合
        SBJs.update(idx for idx, arc in enumerate(arcs) if
                    arc.relation == 'SBV' and arc.head - 1 in HEDs and abs(arc.head - 1 - idx) == 1)
        # 若是从SBJ发出的的SBV或者COO关系，则将其加入 SBJs 集合
        SBJs.update(idx for idx, arc in enumerate(arcs) if (arc.relation == 'SBV' or arc.relation == 'COO')
                    and arc.head - 1 in SBJs)
        # 若其词性为 r 且存在于疑问词表，则将其加入 QWs 集合
        QWs.update(idx for idx in range(len(words)) if postags[idx] == 'r' and words[idx] in que_words)
        # 将疑问词的父节点加入 RELs 集合
        # RELs.update(idx for idx, arc in enumerate(arcs) if arc.head - 1 in QWs)
        RELs.update(arcs[idx].head - 1 for idx in QWs)

        return HEDs, SBJs, QWs, RELs

    def extract(self, text):
        words, postags, netags, arcs = self.ltp.get_all(text)

        HEDs, SBJs, QWs, RELs = self.ccw(words, postags, arcs)

        ccw_features = ['key-' + '/'.join([words[i], postags[i], arcs[i].relation]) for i in HEDs | SBJs | QWs | RELs]
        key_set = HEDs | SBJs | QWs | RELs
        ccw_features += ['key-' + '/'.join([words[i], postags[i]]) for i in key_set]
        ccw_features += ['key-' + '/'.join([words[i], netags[i]]) for i in key_set]
        ccw_features += ['key-' + '/'.join([postags[i], netags[i]]) for i in key_set]
        ccw_features += ['key-' + words[i] for i in key_set]

        # # 核心词的近义词
        # syn_features = []
        # for i in HEDs | SBJs:
        #     syn_words, score = synonyms.nearby(str(words[i]))
        #     for idx, w in enumerate(syn_words):
        #         if score[idx] < 0.6:
        #             break
        #         syn_features.append(w)

        features = ['/'.join([words[i], postags[i]]) for i in range(len(words))]
        features += ['/'.join([words[i], netags[i]]) for i in range(len(words))]
        features += ['/'.join([words[i], postags[i], netags[i]]) for i in range(len(words))]

        # unigram
        # (0.8706240487062404, 0.743531202435312)
        # return list(words)

        # unigram & bigram
        # (0.880517503805175, 0.7557077625570776)
        # return list(words) + ['/'.join(x) for x in zip(words[:-1], words[1:])]

        # unigram + 近义词
        # (0.8698630136986302, 0.7572298325722984)
        # return list(words) + syn_features

        # w/p 组合特征
        # (0.8713850837138508, 0.7427701674277016)
        # return ['/'.join([words[i], postags[i]]) for i in range(len(words))]

        # w/p + w/n + w/p/n 组合特征
        # (0.8850837138508372, 0.7686453576864536)
        # return features

        # unigram + w/p + w/n + w/p/n
        # (0.8873668188736682, 0.7770167427701674)
        # return list(words) + features

        # unigram + w/p + w/n + w/p/n + 问题类别线索词特征
        # (0.9033485540334856, 0.8112633181126332)
        return list(words) + features + ccw_features

        # unigram+bigram + w/p + w/n + w/p/n + 问题类别线索词特征
        # (0.9033485540334856, 0.8105022831050228)
        # return list(words) + ['/'.join(x) for x in zip(words[:-1], words[1:])] + features + ccw_features

        # 复现论文《融合类别线索词的中文问题分类》 分类器和疑问词表有差别
        # (0.8995433789954338, 0.7945205479452054)
        # feature1 = ['/'.join([words[i], postags[i], netags[i]]) for i in range(len(words))]
        # feature1 += [words[i] for i in HEDs | SBJs | QWs | RELs]
        # feature1 += ['/'.join([words[i], postags[i]]) for i in RELs]
        # return feature1


def get_data(path):
    texts, labels, subtype_label = [], [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            content = line.strip().split('\t')
            if len(line) < 3:
                continue
            labels.append(content[0])
            subtype_label.append(content[1])
            texts.append(content[2])

    return texts, labels, subtype_label


if __name__ == '__main__':
    fe = QuestionClassifier(load=False)
    fe.train(fit_test=False)

    # fe = QuestionClassifier()

    print(fe.predict())  # (0.9025875190258752, 0.8112633181126332)
    #
    s = '降雨最少的大陆是哪块'
    print(fe.predict_sent(s))

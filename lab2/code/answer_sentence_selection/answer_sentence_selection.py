#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/7
"""

import os
import json
import numpy as np
from scipy.linalg import norm
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from gensim.summarization.bm25 import BM25
import gensim
import distance

from code.answer_sentence_selection.ltp import LTP
from code.question_classification.bow import BOW
from code.question_classification.question_classification import QuestionClassifier

DIR_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'data')
MODEL_PATH = os.path.join(DIR_PATH, os.pardir, os.pardir, 'model')


class FeatureExtractor(object):
    def __init__(self):
        self.ltp = LTP()
        self.d2v_model = gensim.models.doc2vec.Doc2Vec.load(os.path.join(DIR_PATH, 'd2v.model'))  # 训练好的doc2vec模型
        self.real_word_pos = ['n', 'nd', 'nh', 'ni', 'nl', 'ns', 'nt', 'nz', 'v', 'a', 'm']  # 实词词性:名次、动词、形容词、数
        self.pos_bow = BOW()
        self.pos_bow.fit([['a', 'b', 'c', 'd', 'e', 'g', 'h', 'i', 'j', 'k', 'm', 'n', 'nd', 'nh', 'ni', 'nl', 'ns',
                           'nt', 'nz', 'o', 'p', 'q', 'r', 'u', 'v', 'wp', 'ws', 'x', 'z']])
        self.qc = QuestionClassifier()
        self.qc.load()
        with open(os.path.join(DIR_PATH, 'data.json'), 'r', encoding='utf-8') as f:
            self.preprocessed_data = [json.loads(x) for x in f.readlines()]

    def self_feature(self, doc, corpus):  # 提取答句自身特征
        new_feature = []
        for i, sent in enumerate(doc):
            seg = corpus[i]
            pos = self.ltp.pos(seg)
            ne = self.ltp.recognize(seg, pos)

            sent_feature = []
            sent_feature.append(1) if ':' in sent or '：' in sent else sent_feature.append(0)  # 是否包含冒号
            sent_feature.append(len(seg))  # 句子中词数
            sent_feature.append(len([x for x in pos if x in self.real_word_pos]))  # 句子中实词数：名词、形容词、动词、数
            sent_feature += self.pos_bow.transform([pos])[0]  # 词性
            sent_feature.append(sum([1 if x == 'S-Nh' or x == 'B-Nh' else 0 for x in ne]))  # 人名
            sent_feature.append(sum([1 if x == 'S-Ni' or x == 'B-Ni' else 0 for x in ne]))  # 地名
            sent_feature.append(sum([1 if x == 'S-Ns' or x == 'B-Ns' else 0 for x in ne]))  # 机构名
            new_feature.append(sent_feature)
        return new_feature

    @staticmethod
    def similarity_feature(corpus):  # 提取答案句与问句之间的相似特征

        def calc_sim(vec):  # 计算余弦相似度
            sim = []
            for i in range(len(corpus) - 1):
                denominator = norm(vec[i]) * norm(vec[-1])
                val = np.dot(vec[i], vec[-1]) / denominator if denominator != 0 else 0
                sim.append(val)
            return sim

        # one-hot
        bow = BOW()
        bow.fit(corpus)
        onehot_vec = bow.transform(corpus)
        # TF
        corpus_join = [' '.join(sent) for sent in corpus]
        count_vectorizer = CountVectorizer()
        count_vec = count_vectorizer.fit_transform(corpus_join).toarray()
        # TF-IDF
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_vec = tfidf_vectorizer.fit_transform(corpus_join).toarray()

        onehot_sim = calc_sim(onehot_vec)  # one-hot 相似度
        tf_sim = calc_sim(count_vec)  # tf 相似度
        tfidf_sim = calc_sim(tfidf_vec)  # tf-idf 相似度
        return tf_sim, tfidf_sim, onehot_sim

    @staticmethod
    def ngram_feature(corpus):  # unigram 词共现比例 & bigram 词共现比例
        unigram = corpus
        bigram = [['/'.join(x) for x in zip(sent[:-1], sent[1:])] for sent in corpus]

        unigram_ratio, bigram_ratio = [], []
        que_uni, que_bi = set(unigram[-1]), set(bigram[-1])
        for i in range(len(corpus) - 1):
            sent_uni, sent_bi = set(unigram[i]), set(bigram[i])
            r1 = len(sent_uni & que_uni) / len(sent_uni | que_uni)
            r2 = len(sent_bi & que_bi) / len(sent_bi | que_bi)
            unigram_ratio.append(r1)  # unigram 词共现比例
            bigram_ratio.append(r2)  # bigram 词共现比例
        return unigram_ratio, bigram_ratio

    def levenshtein(self, document, question):  # 编辑距离
        result = [distance.levenshtein(question, sent) for sent in document]
        m = max(result)
        # return [x / m for x in result]
        return result

    # 提取特征
    def extract(self):
        svm_rank_train = []
        for qid, data in enumerate(self.preprocessed_data):
            if qid % 100 == 0:
                print(qid)
            document = data['document']
            answer_idx = data['answer_idx']
            question = data['question']

            corpus = document + [question]  # corpus[-1]：que ；corpus[:-1]：doc
            corpus = [list(self.ltp.seg(sent)) for sent in corpus]

            sent_feature = self.self_feature(document, corpus)  # 答句本身特征
            unigram_ratio, bigram_ratio = self.ngram_feature(corpus)  # unigram 词共现比例 & bigram 词共现比例
            tf_sim, tfidf_sim, onehot_sim = self.similarity_feature(corpus)  # 答句s 与 问句 的相似度
            bm25 = BM25(corpus[:-1])
            bm25_scores = bm25.get_scores(corpus[-1])  # BM25评分
            levenshtein_dis = self.levenshtein(document, question)  # 编辑距离
            c1, c2 = self.qc.predict_sent(question)  # 问题分类
            que_d2v_tag = '-'.join([str(qid), str(len(document))])  # 问题的doc2vec tag

            for sent_idx, sent in enumerate(document):
                features = []
                features += sent_feature[sent_idx]
                features.append(unigram_ratio[sent_idx])
                features.append(bigram_ratio[sent_idx])
                features.append(tf_sim[sent_idx])
                features.append(tfidf_sim[sent_idx])
                features.append(onehot_sim[sent_idx])
                features.append(bm25_scores[sent_idx])
                features.append(levenshtein_dis[sent_idx])
                # features.append(c1)
                # features.append(c2)

                # sent_d2v_tag = '-'.join([str(qid), str(sent_idx)])
                # d2v_sim = self.d2v_model.docvecs.similarity(sent_d2v_tag, que_d2v_tag)
                # features.append(d2v_sim)  # 9
                # features += (self.d2v_model.docvecs[sent_d2v_tag] - self.d2v_model.docvecs[que_d2v_tag]).tolist()

                s = '1' if sent_idx in answer_idx else '0'  # label
                s += ' qid:' + str(qid) + ' '  # qid
                s += ' '.join([str(k + 1) + ':' + str(f) for k, f in enumerate(features)])  # feature
                svm_rank_train.append(s)

        with open(os.path.join(MODEL_PATH, 'svm_rank', 'svmrank_train.dat'), 'w', encoding='utf-8')as f:
            f.write('\n'.join(svm_rank_train))


if __name__ == '__main__':
    fe = FeatureExtractor()
    fe.extract()

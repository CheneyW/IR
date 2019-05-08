#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@author : 王晨懿
@studentID : 1162100102
@time : 2019/5/8
"""

import os
from pyltp import Segmentor
from sklearn import preprocessing, linear_model, naive_bayes, metrics
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import pickle

DATA_PATH = os.path.join(os.pardir, os.pardir, 'data')
MODEL_PATH = os.path.join(os.pardir, os.pardir, 'model')

segmentor = Segmentor()  # 初始化实例
segmentor.load(os.path.join(MODEL_PATH, 'ltp', 'cws.model'))  # 加载模型


def get_data(path, subtype=False):
    labels, texts = [], []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            content = line.split()
            texts.append(content[1])
            if subtype:
                labels.append(content[0])
            else:
                labels.append(content[0][:3])

    texts = [' '.join(list(segmentor.segment(sent))) for sent in texts]
    return texts, labels


# 问题分类器
class QuestionClassifier(object):
    def __init__(self, fit_test=False, subtype=False):
        """
        :param fit_test: True: 将测试集一同训练 ; False: 只训练训练集
        :param subtype: True: label取子类 ; False: label取大类
        """
        self.fit_test = fit_test
        self.subtype = subtype
        if self.subtype:
            self.vectorizer_path = os.path.join(MODEL_PATH, 'ques_clf', 'vectorizer_subtype.pkl')
            self.label_encoder_path = os.path.join(MODEL_PATH, 'ques_clf', 'label_encoder_subtype.pkl')
            self.model_path = os.path.join(MODEL_PATH, 'ques_clf', 'model_subtype.pkl')
        else:
            self.vectorizer_path = os.path.join(MODEL_PATH, 'ques_clf', 'vectorizer.pkl')
            self.label_encoder_path = os.path.join(MODEL_PATH, 'ques_clf', 'label_encoder.pkl')
            self.model_path = os.path.join(MODEL_PATH, 'ques_clf', 'model.pkl')
        self.fit()

    def fit(self):
        train_x, train_y = get_data('trian_questions.txt', subtype=self.subtype)
        if self.fit_test:
            test_x, test_y = get_data('test_questions.txt', subtype=self.subtype)
            train_x += test_x
            train_y += test_y
        # 词袋模型
        vectorizer = CountVectorizer()
        train_x = vectorizer.fit_transform(train_x)
        # label编码为目标变量
        label_encoder = preprocessing.LabelEncoder()
        train_y = label_encoder.fit_transform(train_y)
        # 建立模型
        model = linear_model.LogisticRegression()
        model.fit(train_x, train_y)
        # save
        with open(self.vectorizer_path, 'wb') as fw:
            pickle.dump(vectorizer, fw)
        with open(self.label_encoder_path, 'wb') as fw:
            pickle.dump(label_encoder, fw)
        with open(self.model_path, 'wb') as fw:
            pickle.dump(model, fw)

    # 读取模型文件并预测
    def predict(self, test_x):
        vectorizer = pickle.load(open(self.vectorizer_path, "rb"))
        test_x = vectorizer.transform(test_x)
        model = pickle.load(open(self.model_path, "rb"))
        return model.predict(test_x)

    def evaluate(self, predictions, test_y):
        label_encoder = pickle.load(open(self.label_encoder_path, "rb"))
        test_y = label_encoder.transform(test_y)
        return metrics.accuracy_score(predictions, test_y)


# 测试模型效果
def test_perform():
    def train_model(model, train_x, train_y, test_x, test_y):
        model.fit(train_x, train_y)
        predictions = model.predict(test_x)
        accuracy = metrics.accuracy_score(predictions, test_y)
        return accuracy

    subtype = True
    train_x, train_y = get_data('trian_questions.txt', subtype=subtype)
    test_x, test_y = get_data('test_questions.txt', subtype=subtype)
    # 词袋模型
    vectorizer = CountVectorizer()
    train_x = vectorizer.fit_transform(train_x)
    test_x = vectorizer.transform(test_x)
    print(train_x.toarray().shape)
    # label编码为目标变量
    label_encoder = preprocessing.LabelEncoder()
    train_y = label_encoder.fit_transform(train_y)
    test_y = label_encoder.transform(test_y)
    # 建立模型
    model = linear_model.LogisticRegression()
    # model = naive_bayes.MultinomialNB()
    accuracy = train_model(model, train_x, train_y, test_x, test_y)
    print(accuracy)


if __name__ == '__main__':
    test_perform()

    # clf = QuestionClassifier(fit_test=False, subtype=False)
    # test_x, test_y = get_data('test_questions.txt', subtype=False)
    # predictions = clf.predict(test_x)
    # print(clf.evaluate(predictions, test_y))

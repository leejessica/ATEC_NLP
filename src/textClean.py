# -*- coding: utf-8 -*-

import re
import pandas as pd
import os

baseDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
dataDir = baseDir + '/data'


class TextClean(object):

    def __init__(self):
        pass

    @staticmethod
    def textClean(text):
        wrong_words = ['零时', 'ma', '怎嚒 ', '怎怎么', '对酒', '花贝', '花被', '借贝', '借被', '接清', '一支', '从新', '怎么怎么', '生份证',
                       '?_?', '余利宝', '话呗', '胃啥', '用这用这', '没发交', '海款', '借代', '换清', 'qb', '扣币', '受权', '压金', 'yue', '如和',
                       '还玩', '為什麼', '冲话费', '登陆', '届不了', '撒意思', '整么', 'qq币', '俩千']
        right_words = ['临时', '吗', '怎么', '怎么', '多久', '花呗', '花呗', '借呗', '借呗', '结清', '一致', '重新', '怎么', '身份证',
                       '?', '余额宝', '花呗', '为啥', '用着用着', '没法交', '还款', '借贷', '还清', 'q币', 'q币', '授权', '押金', '余额', '如何',
                       '还完', '为什么', '充话费', '登录', '借不了', '啥意思', '怎么', 'q币', '两千']

        for w1, w2 in zip(wrong_words, right_words):
            text = text.replace(w1, w2)
        return text

    @staticmethod
    def rm_punctuation(text):
        text = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。：？?、~@#￥%……&*（）]+".decode("utf8"), "".decode("utf8"),
                      text)
        return text


if __name__ == "__main__":
    train_data = pd.read_table("../data/raw_data/atec_nlp_sim_train.csv", header=None, encoding='utf_8_sig')
    train_data.rename(columns={0: "ID", 1: "Sentence1", 2: "Sentence2", 3: "label"}, inplace=True)
    for col in ["Sentence1", "Sentence2"]:
        train_data[col] = train_data[col].apply(lambda x: TextClean.textClean(x))
    train_data.to_csv('../data/raw_data/atec_nlp_sim_train_processed.csv',
                      columns=['ID', 'Sentence1', 'Sentence2', 'label'], encoding='utf-8', )

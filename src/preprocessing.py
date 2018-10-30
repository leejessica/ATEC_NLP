# -*- coding: utf-8 -*-
import pandas as pd
import jieba
import os
from gensim.models import Word2Vec
import numpy as np
from gensim import corpora
from keras.preprocessing.sequence import pad_sequences
import itertools

from src.textClean import TextClean

baseDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
dataDir = baseDir + '/data'


def tokenizer_word(sentence, userdict):
    jieba.load_userdict(userdict)
    sentence = TextClean.rm_punctuation(sentence)
    return jieba.lcut(sentence)


def cut_word(data, userdict):
    for col in ['Sentence1', 'Sentence2']:
        data[col + '_cut'] = data[col].apply(lambda s: tokenizer_word(s, userdict))
    return data


def load_data(filename):
    data = pd.read_csv(filename)
    data = cut_word(data)
    return data


# convert the wv word vectors into a numpy matrix that is suitable for insertion
def get_embedding_index(modelname):
    w2vModel = Word2Vec.load(dataDir + '/extra_dict/' + modelname)
    embedding_index = np.zeros((len(w2vModel.wv.vocab), w2vModel.wv.vector_size))
    for i in range(len(w2vModel.wv.vocab)):
        embedding_vector = w2vModel.wv[w2vModel.wv.index2word[i]]
        if embedding_vector is not None:
            embedding_index[i] = embedding_vector
    return embedding_index, w2vModel.wv.vector_size


def pad_sequence(max_length, train_data):
    x = {'left': train_data['Sentence1_w2v'], 'right': train_data['Sentence2_w2v']}
    for dataset, side in itertools.product([x], ['left', 'right']):
        dataset[side] = pad_sequences(dataset[side], padding='pre', truncating='post', maxlen=max_length)
    return dataset


def train_test_split(X, mask, Y, frac=0.8):
    n_samples = len(Y)
    idx = np.random.permutation(n_samples)
    train_idx = idx[:int(n_samples * frac)]
    test_idx = [i for i in idx if i not in train_idx]
    train_X = {}
    train_mask = {}
    test_X = {}
    test_mask = {}
    for side, value in X.items():
        sents1 = [value[i] for i in train_idx]
        sents2 = [value[i] for i in test_idx]
        train_X[side] = sents1
        test_X[side] = sents2
        mask1 = [mask[side][i] for i in train_idx]
        mask2 = [mask[side][i] for i in test_idx]
        train_mask[side] = mask1
        test_mask[side] = mask2
    train_Y = [Y[i] for i in train_idx]
    test_Y = [Y[i] for i in test_idx]

    return [train_X, train_mask, train_Y], [test_X, test_mask, test_Y]


def get_batch_data(data, start_idx, batch_size):
    """

    :param data:  list of data, [X, mask, Y]
    :param start_idx:
    :param batch_size:
    :return:
    """
    batch_X={}
    batch_mask={}
    batch_Y=[]
    if (start_idx+batch_size)<=len(data):
        end_idx=start_idx+batch_size
    else:
        end_idx=len(data[2])
        start_idx=end_idx-batch_size
    for side in ['left', 'right']:
        batch_X[side]=data[0][side][start_idx:end_idx]
        batch_mask[side]=data[1][side][start_idx:end_idx]
    return batch_X, batch_mask, batch_Y


class My_Vocabulary(object):
    def __init__(self, data, embedding_index, embed_dim, sentence_maxlen):
        self.data = data
        self.embedding_index = embedding_index
        self.embed_dim = embed_dim
        self.sentence_maxlen = sentence_maxlen

    def get_vocabulary(self):
        train_list = self.data['Sentence1_cut'].tolist() + self.data['Sentence2_cut'].tolist()
        dictionary = corpora.Dictionary(train_list)
        self.vocab = dictionary.token2id
        return self.vocab

    def get_embedding_matrix(self):
        """

        :return:
        """
        all_embs = np.stack(list(self.embedding_index.values()))
        embs_mean, embs_std = all_embs.mean(), all_embs.std()
        nb_words = len(self.vocab)
        self.embedding_matrix = np.random.normal(embs_mean, embs_std, (nb_words + 1, self.embed_dim))
        for word, id in self.vocab.items():
            if word in self.embedding_index:
                embedding_vector = self.embedding_index[word]
                if embedding_vector is not None:
                    self.embedding_matrix[id + 1] = embedding_vector
        self.embedding_matrix[0] = np.zeros(shape=(1, self.embed_dim))
        return self.embedding_matrix

    def word_ids(self, wordlist):
        id_list = []
        for word in wordlist:
            if word in self.vocab:
                id_list.append(self.vocab[word])
        return id_list

    def embedding_data(self):
        for col in ['Sentence1_cut', 'Sentence2_cut']:
            self.data[col[:-3] + '_idlist'] = self.data[col].apply(lambda x: self.sentence2id(x))
        return self.data

    def padding_and_masking(self):
        X = {'left': self.data['Sentence1_idlist'].tolist(), 'right': self.data['Sentence2_idlist'].tolist()}
        mask = {'left': np.zeros(shape=(len(self.data), self.sentence_maxlen), dtype=np.int16),
                'right': np.zeros(shape=(len(self.data), self.sentence_maxlen), dtype=np.int16)}
        Y = self.data['label'].tolist()
        for side, value in X.items():
            for i, sentences in enumerate(value):
                if len(sentences) <= self.sentence_maxlen:
                    mask[side][i][:len(sentences)] = 1
                else:
                    mask[side][i][:] = 1
            X[side] = pad_sequences(X[side], padding='post', truncating='post', maxlen=self.sentence_maxlen).tolist()

        return X, mask, Y

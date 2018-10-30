# -*- coding: utf-8 -*-
import logging
import tensorflow as tf
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def create_single_RNN(x, embedding_matrix, num_units, scope, cell='LSTM'):
    """

    :param x: ID list of sentence, shape=(batch_size, sentence_length)
    :param embedding_matrix: embedding vector of all the words in our vocab
    :param embed_size: the size of embedding vector
    :param max_length: max length of sentence
    :param cell: LSTM or GRU
    :return:
    """
    with tf.name_scope("Embedding_Layer"):
        myword2vec = tf.get_variable(name="myword2vec", initializer=embedding_matrix,
                                     shape=[embedding_matrix.shape()[0], embedding_matrix.shape()[1]], trainable=False)
        embedded_sentences = tf.nn.embedding_lookup(myword2vec, x)

    with tf.name_scope("RNN_Cell_Layer"):
        if cell == 'LSTM':
            with tf.variable_scope('LSTM_' + scope, reuse=tf.AUTO_REUSE, dtype=tf.float32):
                used_cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units, initializer=tf.initializers.orthogonal,
                                                    reuse=tf.get_variable_scope().reuse)
        if cell == 'GRU':
            with tf.variable_scope('GRU_' + scope, reuse=tf.AUTO_REUSE, dtype=tf.float32):
                used_cell = tf.nn.rnn_cell.GRUCell(num_units=num_units, kernel_initializer=tf.initializers.orthogonal,
                                                   bias_initializer=tf.initializers.random_uniform,
                                                   reuse=tf.get_variable_scope().reuse)
        output, state = tf.nn.dynamic_rnn(used_cell, embedded_sentences, dtype=tf.float32)
    return output, state


class Siamese_RNN(object):
    def __init__(self, embedding_matrix, num_units, cell, cost_method=None, dist=None, threshold=0.5, **kwargs):
        self.embedding_matrix = embedding_matrix
        self.num_units = num_units
        self.cell = cell
        self.cost_method = cost_method
        self.dist = dist
        self.max_length = kwargs.get('max_length')
        #self.keep_final = kwargs.get('keep_final')
        self.summaries = kwargs.get("summaries", True)
        # time_major=False(default), inputs.shape=[batch_size, max_time, ...]
        self.x1 = tf.placeholder(shape=[None, self.max_length], dtype=tf.float32,
                                 name='input1')
        self.x2 = tf.placeholder(shape=[None, self.max_length], dtype=tf.float32,
                                 name='input2')
        self.mask1 = tf.placeholder(shape=[None, self.max_length], name='mask1')
        self.mask2 = tf.placeholder(shape=[None, self.max_length], name='mask2')
        self.y = tf.placeholder(shape=[None], dtype=tf.int32, name='label')
        self.output1, _ = create_single_RNN(self.x1, self.embedding_matrix, self.num_units, 'side1', self.cell)
        self.output2, _ = create_single_RNN(self.x2, self.embedding_matrix, self.num_units, 'side1', self.cell)

        with tf.name_scope('Sentence_Layer'):
            self.sent1 = tf.reduce_sum(self.output1 * self.mask1[:, :, None], axis=1)
            self.sent2 = tf.reduce_sum(self.output2 * self.mask2[:, :, None], axis=1)

        with tf.name_scope('Cost'):
            if self.cost_method == None:
                if self.dist == None:
                    self.cost = self.contrastive_loss()
                else:
                    self.cost = self.contrastive_loss(self.dist)

        with tf.name_scope('Prediction'):
            if self.dist == None:
                self.distance = self.mahattan_distance()
            if self.dist == 'euclidean':
                self.distance = self.euclidean_distance()

            self.prediction = np.where(self.distance >= threshold, 1, 0)

        with tf.name_scope('Results'):
            _, self.accuracy = tf.metrics.accuracy(labels=self.Y, predictions=self.prediction)
            _, self.recall = tf.metrics.recall(labels=self.Y, predictions=self.prediction)
            self.F1_score = 2 * self.accuracy * self.recall / (self.accuracy + self.recall)

    def euclidean_distance(self):
        diff = tf.sqrt(tf.reduce_sum(tf.square(tf.substr(self.sent1, self.sent2)), axis=1))
        distance = tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)
        return distance

    def mahattan_distance(self):
        diff = tf.reduce_sum(tf.abs(tf.substr(self.sent1, self.sent2)), axis=1)
        distance = tf.clip_by_value(tf.exp(-1.0 * diff), 1e-7, 1.0 - 1e-7)
        return distance

    def contrastive_loss(self, dist='mahattan'):
        if dist == 'mahattan':
            distance = self.mahattan_distance()
        if dist == 'euclidean':
            distance = self.euclidean_distance()
        sum_loss = tf.reduce_sum(self.y * tf.square(distance) + (1.0 - self.y) * tf.square(tf.maximum(1 - distance, 0)),
                                 axis=0)
        normalize_loss = tf.divide(sum_loss, tf.shape(self.y)[0] * 2)
        return normalize_loss

    def save(self, sess, model_path):
        """

        :param sess: current session
        :param model_path: path to file system location
        :return:
        """
        saver = tf.train.Saver()
        save_path = saver.save(sess, model_path)
        return save_path

    def restore(self, sess, model_path):
        """

        :param sess: current session
        :param model_path: path to file system location
        :return:
        """
        meta_path = os.path.join(model_path, '.meta')
        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, model_path)
        logging.info("Model restored from file: %s" % model_path)

    def predict(self, model_path, test_X, test_mask, threshold):
        """

        :param model_path: Path to file system location
        :param x1_test, x2_test:  Data to predict on, shape=[batch_size, sentence_length]
        :return:
        """
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            # Initialize variables
            sess.run(init)
            # Restore model weights and model graph definition from previously saved model
            self.restore(sess, model_path)

            y_dummy = np.empty((len(test_X['left'])))

            distance = sess.run(self.distance,
                                feed_dict={self.x1: test_X['left'],
                                           self.x2: test_X['right'],
                                           self.mask1: test_mask['left'],
                                           self.mask2: test_mask['right'],
                                           self.y: y_dummy})
            result = np.where(distance >= threshold, 1, 0)
        return result

